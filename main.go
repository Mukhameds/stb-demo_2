package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

type Action int

const (
	BUY Action = iota
	WAIT
	RUN
)

func (a Action) String() string {
	switch a {
	case BUY:
		return "BUY"
	case WAIT:
		return "WAIT"
	case RUN:
		return "RUN"
	default:
		return "?"
	}
}

type Emotion int

const (
	CALM Emotion = iota
	FEAR
)

func (e Emotion) String() string {
	if e == FEAR {
		return "fear"
	}
	return "calm"
}

type Bucket int

const (
	LOW Bucket = iota
	MED
	HIGH
)

func (b Bucket) String() string {
	switch b {
	case LOW:
		return "LOW"
	case MED:
		return "MED"
	case HIGH:
		return "HIGH"
	default:
		return "?"
	}
}

func bucketOfRisk(r float64) Bucket {
	if r < 0.33 {
		return LOW
	}
	if r < 0.66 {
		return MED
	}
	return HIGH
}

type Ctx struct {
	TrueRisk float64
	Emotion  Emotion
	Obs1     float64
	Rel1     float64
	Obs2     float64
	Rel2     float64
}

type Toggle struct {
	Goal    bool
	Emotion bool
	Memory  bool
	Inst    bool
	Logic   bool
	Will    bool
}

type Gains struct {
	Goal    float64
	Emotion float64
	Memory  float64
	Inst    float64
	Logic   float64
	Will    float64
}

type Floors struct {
	Emotion float64
	Memory  float64
	Inst    float64
	Logic   float64
	Will    float64
}

// --- Policy mode (training vs eval vs do-check) ---
type PolicyMode struct {
	Deterministic bool    // if true => argmax
	Temperature   float64 // for softmax (training)
	Epsilon       float64 // optional exploration (kept 0 here, but safe)
}

func TrainMode() PolicyMode  { return PolicyMode{Deterministic: false, Temperature: 1.0, Epsilon: 0.0} }
func EvalGreedy() PolicyMode { return PolicyMode{Deterministic: true, Temperature: 1.0, Epsilon: 0.0} } // temperature irrelevant when deterministic

type Memory struct {
	// AVOID-ONLY memory:
	// key = bucket|emotion|action or bucket|any|action
	store map[string]float64 // positive means "avoid penalty"
	cap   float64            // saturation cap
	decay float64            // exponential decay per episode
}

func NewMemory(cap float64, decay float64) *Memory {
	return &Memory{
		store: make(map[string]float64),
		cap:   cap,
		decay: decay,
	}
}

func (m *Memory) DecayAll() {
	if m.decay >= 1.0 {
		return
	}
	for k, v := range m.store {
		nv := v * m.decay
		// small cleanup
		if nv < 1e-6 {
			delete(m.store, k)
			continue
		}
		m.store[k] = nv
	}
}

func (m *Memory) Add(key string, delta float64) {
	m.store[key] += delta
	if m.store[key] > m.cap {
		m.store[key] = m.cap
	}
	if m.store[key] < 0 {
		m.store[key] = 0
	}
}

func memKey(bucket Bucket, emo Emotion, any bool, a Action) string {
	if any {
		return fmt.Sprintf("%s|any|%s", bucket.String(), a.String())
	}
	return fmt.Sprintf("%s|%s|%s", bucket.String(), emo.String(), a.String())
}

func clamp(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

func softmax(logits map[Action]float64, temp float64) map[Action]float64 {
	if temp <= 1e-9 {
		temp = 1e-9
	}
	// numerically stable
	maxv := -1e9
	for _, v := range logits {
		if v > maxv {
			maxv = v
		}
	}
	sum := 0.0
	out := map[Action]float64{}
	for a, v := range logits {
		ev := math.Exp((v - maxv) / temp)
		out[a] = ev
		sum += ev
	}
	for a := range out {
		out[a] /= sum
	}
	return out
}

func argmax(dist map[Action]float64, acts []Action) Action {
	bestA := acts[0]
	bestV := dist[bestA]
	for _, a := range acts[1:] {
		if dist[a] > bestV {
			bestV = dist[a]
			bestA = a
		}
	}
	return bestA
}

func sample(dist map[Action]float64, rng *rand.Rand, acts []Action) Action {
	x := rng.Float64()
	acc := 0.0
	for _, a := range acts {
		acc += dist[a]
		if x <= acc {
			return a
		}
	}
	return acts[len(acts)-1]
}

func pickAction(dist map[Action]float64, mode PolicyMode, rng *rand.Rand, acts []Action) Action {
	if mode.Deterministic {
		return argmax(dist, acts)
	}
	// epsilon-greedy (kept 0 in our modes; still here if you want later)
	if mode.Epsilon > 0 && rng.Float64() < mode.Epsilon {
		return acts[rng.Intn(len(acts))]
	}
	return sample(dist, rng, acts)
}

// --- Environment / reward ---
//
// TrueRisk determines optimal final action:
//   HIGH => RUN best
//   LOW/MED => BUY best
//
// WAIT costs small time penalty but improves estimate quality (by providing obs2).
func reward(trueRisk float64, final Action, usedWait bool) (r float64, catastrophic bool) {
	b := bucketOfRisk(trueRisk)
	optimal := BUY
	if b == HIGH {
		optimal = RUN
	}
	base := 0.0
	if final == optimal {
		base = 0.95
	} else {
		base = 0.30
	}
	// catastrophic penalty: BUY when HIGH
	if b == HIGH && final == BUY {
		base = -1.20
		catastrophic = true
	}
	// wait cost; cancel it if WAIT saved you from a catastrophe (HIGH -> RUN)
	if usedWait {
		base -= 0.06
		if b == HIGH && final == RUN {
			base += 0.06
		}
	}
	return base, catastrophic
}

// --- VOI: value-of-information for WAIT ---
//
// We want VOI high when reliability is low AND risk estimate is near the decision boundary.
// Boundary is around 0.66 (MED/HIGH).
func computeVOI(est float64, rel float64) float64 {
	boundary := 0.66
	// closeness to boundary in [0..1]
	close := 1.0 - clamp(math.Abs(est-boundary)/0.34, 0, 1)
	uncertainty := (1.0 - rel) * close
	// scale to [0..1]
	voi := clamp(uncertainty*1.35, 0, 1)
	return voi
}

type StepResult struct {
	Action   Action
	Dist     map[Action]float64
	Contrib  map[string]float64 // module -> contribution to chosen action
	Trace    string
	VOI      float64
	WaitLogit float64
}

// Score model: modules contribute to BUY/RUN; WAIT is ONLY Logic via VOI.
//
// IMPORTANT: to make causal necessity measurable,
// WAIT_logit must go to ~0 if Logic disabled.
func step1Policy(ctx Ctx, g Gains, f Floors, mem *Memory, tog Toggle, mode PolicyMode, rng *rand.Rand) StepResult {
	b := bucketOfRisk(ctx.Obs1)
	// floors applied
	emGain := math.Max(f.Emotion, g.Emotion)
	memGain := math.Max(f.Memory, g.Memory)
	inGain := math.Max(f.Inst, g.Inst)
	loGain := math.Max(f.Logic, g.Logic)
	wlGain := math.Max(f.Will, g.Will)

	// Hard toggles: if disabled -> 0 gain (do() truly removes)
	if !tog.Emotion {
		emGain = 0
	}
	if !tog.Memory {
		memGain = 0
	}
	if !tog.Inst {
		inGain = 0
	}
	if !tog.Logic {
		loGain = 0
	}
	if !tog.Will {
		wlGain = 0
	}

	// GOAL is frozen; if disabled then 0
	goal := 1.20
	if !tog.Goal {
		goal = 0
	}

	// --- base tendencies ---
	// Instinct: prefers RUN in HIGH estimate, BUY in LOW/MED estimate a bit.
	instBUY := 0.30
	instRUN := 0.30
	if b == HIGH {
		instRUN = 1.10
		instBUY = 0.10
	} else if b == LOW {
		instBUY = 1.00
		instRUN = 0.15
	} else { // MED
		instBUY = 0.65
		instRUN = 0.45
	}

	// Emotion: FEAR pushes RUN, CALM pushes BUY slightly.
	emoBUY := 0.20
	emoRUN := 0.20
	if ctx.Emotion == FEAR {
		emoRUN = 0.65
		emoBUY = 0.10
	} else {
		emoBUY = 0.35
		emoRUN = 0.10
	}

	// Will: a stabilizer – prefers acting (BUY/RUN) rather than WAIT,
	// but NOT huge.
	willAct := 0.45

	// Memory AVOID-ONLY penalty (never boosts):
	// Scientific-friendly: memory is SECONDARY and only penalizes BUY in MED/HIGH.
	// In LOW bucket it should not interfere.
	memPenBUY := 0.0
	if tog.Memory && memGain > 0 {
		if b == HIGH || b == MED {
			k1 := memKey(b, ctx.Emotion, false, BUY)
			k2 := memKey(b, ctx.Emotion, true, BUY)
			raw := mem.store[k1] + mem.store[k2]
			// MED gets weaker trauma weight than HIGH
			if b == MED {
				raw *= 0.55
			}
			memPenBUY = clamp(raw, 0, 3.0)
		}
	}

	// --- WAIT: ONLY logic via VOI ---
	voi := computeVOI(ctx.Obs1, ctx.Rel1)
	voiThreshold := 0.18
	voiScale := 2.60
	waitLogit := loGain * math.Max(0, (voi-voiThreshold)) * voiScale

	// BUY/RUN logits (NO VOI inside)
	buyLogit := goal +
		inGain*instBUY +
		emGain*emoBUY +
		wlGain*willAct -
		memGain*memPenBUY

	runLogit := 0.30 + // small baseline
		inGain*instRUN +
		emGain*emoRUN +
		wlGain*willAct

	logits := map[Action]float64{
		BUY:  buyLogit,
		WAIT: waitLogit,
		RUN:  runLogit,
	}

	temp := mode.Temperature
	if mode.Deterministic {
		// temperature irrelevant, but keep stable
		temp = 1.0
	}
	dist := softmax(logits, temp)
	chosen := pickAction(dist, mode, rng, []Action{BUY, WAIT, RUN})

	// contributions to chosen action (for debug)
	contrib := map[string]float64{}
	switch chosen {
	case WAIT:
		contrib["Logic"] = waitLogit
	case BUY:
		contrib["Goal"] = goal
		contrib["Instinct"] = inGain * instBUY
		contrib["Emotion"] = emGain * emoBUY
		contrib["Will"] = wlGain * willAct
		contrib["Memory"] = -memGain * memPenBUY
	case RUN:
		contrib["Instinct"] = inGain * instRUN
		contrib["Emotion"] = emGain * emoRUN
		contrib["Will"] = wlGain * willAct
	}

	return StepResult{
		Action:    chosen,
		Dist:      dist,
		Contrib:   contrib,
		Trace:     "Goal->Emotion->Memory->Instinct->Logic->Will",
		VOI:       voi,
		WaitLogit: waitLogit,
	}
}

func step2Policy(ctx Ctx, g Gains, f Floors, mem *Memory, tog Toggle, mode PolicyMode, rng *rand.Rand) StepResult {
	// Step2 happens only after WAIT, choose BUY vs RUN using obs2 (high reliability).
	b := bucketOfRisk(ctx.Obs2)

	emGain := math.Max(f.Emotion, g.Emotion)
	memGain := math.Max(f.Memory, g.Memory)
	inGain := math.Max(f.Inst, g.Inst)
	wlGain := math.Max(f.Will, g.Will)

	if !tog.Emotion {
		emGain = 0
	}
	if !tog.Memory {
		memGain = 0
	}
	if !tog.Inst {
		inGain = 0
	}
	if !tog.Will {
		wlGain = 0
	}

	goal := 1.20
	if !tog.Goal {
		goal = 0
	}

	instBUY := 0.30
	instRUN := 0.30
	if b == HIGH {
		instRUN = 1.25
		instBUY = 0.05
	} else {
		instBUY = 1.10
		instRUN = 0.15
	}

	emoBUY := 0.15
	emoRUN := 0.15
	if ctx.Emotion == FEAR {
		emoRUN = 0.55
		emoBUY = 0.08
	} else {
		emoBUY = 0.30
		emoRUN = 0.10
	}

	willAct := 0.35

	// Memory penalty ONLY for BUY (also here, step2).
	// Keep it secondary: MED weaker than HIGH; LOW = no penalty.
	memPenBUY := 0.0
	if tog.Memory && memGain > 0 {
		if b == HIGH || b == MED {
			k1 := memKey(b, ctx.Emotion, false, BUY)
			k2 := memKey(b, ctx.Emotion, true, BUY)
			raw := mem.store[k1] + mem.store[k2]
			if b == MED {
				raw *= 0.55
			}
			memPenBUY = clamp(raw, 0, 3.0)
		}
	}

	buyLogit := goal +
		inGain*instBUY +
		emGain*emoBUY +
		wlGain*willAct -
		memGain*memPenBUY

	runLogit := 0.25 +
		inGain*instRUN +
		emGain*emoRUN +
		wlGain*willAct

	// normalize with softmax over two actions
	maxv := math.Max(buyLogit, runLogit)
	eBUY := math.Exp(buyLogit - maxv)
	eRUN := math.Exp(runLogit - maxv)
	sum := eBUY + eRUN

	dist := map[Action]float64{
		BUY:  eBUY / sum,
		RUN:  eRUN / sum,
		WAIT: 0,
	}

	chosen := pickAction(dist, mode, rng, []Action{BUY, RUN})

	contrib := map[string]float64{}
	if chosen == BUY {
		contrib["Goal"] = goal
		contrib["Instinct"] = inGain * instBUY
		contrib["Emotion"] = emGain * emoBUY
		contrib["Will"] = wlGain * willAct
		contrib["Memory"] = -memGain * memPenBUY
	} else {
		contrib["Instinct"] = inGain * instRUN
		contrib["Emotion"] = emGain * emoRUN
		contrib["Will"] = wlGain * willAct
	}

	return StepResult{
		Action:    chosen,
		Dist:      dist,
		Contrib:   contrib,
		Trace:     "Goal->Emotion->Memory->Instinct->Logic->Will",
		VOI:       0,
		WaitLogit: 0,
	}
}

// --- training ---
//
// We do simple REINFORCE-like gain updates + AVOID-ONLY memory writes:
// if an action causes negative outcome (esp catastrophic), increase avoid penalty for that action in that bucket.
func trainEpisode(rng *rand.Rand, g *Gains, f Floors, mem *Memory) (avgR float64) {
	// memory decay once per episode (brain-like forgetting)
	mem.DecayAll()

	// generate true risk
	trueRisk := rng.Float64()
	emo := CALM
	if rng.Float64() < 0.35 {
		emo = FEAR
	}

	// observation model:
	// obs1 has variable reliability; obs2 after WAIT is higher reliability.
	rel1 := clamp(0.25+0.55*rng.Float64(), 0.25, 0.80)
	rel2 := clamp(rel1+0.30+0.20*rng.Float64(), 0.60, 0.98)

	noise1 := (rng.Float64()*2 - 1) * (1.0 - rel1) * 0.55
	noise2 := (rng.Float64()*2 - 1) * (1.0 - rel2) * 0.25
	obs1 := clamp(trueRisk+noise1, 0, 1)
	obs2 := clamp(trueRisk+noise2, 0, 1)

	ctx := Ctx{TrueRisk: trueRisk, Emotion: emo, Obs1: obs1, Rel1: rel1, Obs2: obs2, Rel2: rel2}

	// toggles all on during training
	tog := Toggle{Goal: true, Emotion: true, Memory: true, Inst: true, Logic: true, Will: true}
	mode := TrainMode()

	// step1
	s1 := step1Policy(ctx, *g, f, mem, tog, mode, rng)

	usedWait := false
	final := s1.Action
	if s1.Action == WAIT {
		usedWait = true
		s2 := step2Policy(ctx, *g, f, mem, tog, mode, rng)
		final = s2.Action
	}

	r, catastrophic := reward(trueRisk, final, usedWait)

	// --- AVOID-ONLY memory write (BUY aversion only) ---
	// If catastrophic BUY in HIGH -> strong avoid for BUY in that true bucket.
	tb := bucketOfRisk(trueRisk)

	// only write avoidance for BUY (memory is "do not BUY here")
	if final == BUY && (r < 0.30 || catastrophic) {
		k := memKey(tb, emo, false, BUY)
		kAny := memKey(tb, emo, true, BUY)

		delta := 0.18
		if catastrophic {
			delta = 0.45
		}
		mem.Add(k, delta)
		mem.Add(kAny, 0.10*delta)
	}

	// --- gain updates (gentle) ---
	// We push Logic up when WAIT chosen and outcome improves;
	// push Instinct/Will down a bit if catastrophic.
	lr := 0.03

	if usedWait && r > 0.40 {
		// reward WAIT via Logic gain
		g.Logic += lr * (r - 0.30)
	}
	if !usedWait && catastrophic {
		// discourage blind BUY in danger: increase Memory, reduce Will/Inst
		g.Memory += lr * 0.6
		g.Will -= lr * 0.4
		g.Inst -= lr * 0.2
	}
	// mild stabilization
	g.Emotion += lr * 0.05 * (r - 0.35)
	g.Inst += lr * 0.05 * (r - 0.35)
	g.Will += lr * 0.03 * (r - 0.35)

	// clamp gains
	g.Emotion = clamp(g.Emotion, 0.70, 3.0)
	g.Memory = clamp(g.Memory, 0.70, 3.0)
	g.Inst = clamp(g.Inst, 0.70, 3.0)
	g.Logic = clamp(g.Logic, 0.70, 3.0)
	g.Will = clamp(g.Will, 0.70, 3.0)
	// Goal frozen at 1.0
	g.Goal = 1.0

	return r
}

type StatsRow struct {
	Label   string
	WaitPct int
	BuyPct  int
	RunPct  int
	AvgR    float64
}

// Greedy policy stats (deterministic) — avoids exploration noise.
func policyStatsGreedy(rng *rand.Rand, g Gains, f Floors, mem *Memory, trueBucket Bucket, emo Emotion, n int) StatsRow {
	tog := Toggle{Goal: true, Emotion: true, Memory: true, Inst: true, Logic: true, Will: true}
	mode := EvalGreedy()

	waitC := 0
	buyC := 0
	runC := 0
	sumR := 0.0

	for i := 0; i < n; i++ {
		// sample true risk inside bucket
		var tr float64
		switch trueBucket {
		case LOW:
			tr = rng.Float64() * 0.33
		case MED:
			tr = 0.33 + rng.Float64()*0.33
		case HIGH:
			tr = 0.66 + rng.Float64()*0.34
		}

		// obs model
		rel1 := clamp(0.25+0.55*rng.Float64(), 0.25, 0.80)
		rel2 := clamp(rel1+0.30+0.20*rng.Float64(), 0.60, 0.98)
		noise1 := (rng.Float64()*2 - 1) * (1.0 - rel1) * 0.55
		noise2 := (rng.Float64()*2 - 1) * (1.0 - rel2) * 0.25
		obs1 := clamp(tr+noise1, 0, 1)
		obs2 := clamp(tr+noise2, 0, 1)

		ctx := Ctx{TrueRisk: tr, Emotion: emo, Obs1: obs1, Rel1: rel1, Obs2: obs2, Rel2: rel2}
		s1 := step1Policy(ctx, g, f, mem, tog, mode, rng)

		usedWait := false
		final := s1.Action
		if s1.Action == WAIT {
			usedWait = true
			waitC++
			s2 := step2Policy(ctx, g, f, mem, tog, mode, rng)
			final = s2.Action
		}

		if final == BUY {
			buyC++
		} else {
			runC++
		}
		r, _ := reward(tr, final, usedWait)
		sumR += r
	}

	label := fmt.Sprintf("TRUE_%s  %s", trueBucket.String(), emo.String())
	return StatsRow{
		Label:   label,
		WaitPct: int(math.Round(100.0 * float64(waitC) / float64(n))),
		BuyPct:  int(math.Round(100.0 * float64(buyC) / float64(n))),
		RunPct:  int(math.Round(100.0 * float64(runC) / float64(n))),
		AvgR:    sumR / float64(n),
	}
}

func topMemory(mem *Memory, k int) []string {
	type kv struct{ K string; V float64 }
	var arr []kv
	for kk, vv := range mem.store {
		arr = append(arr, kv{kk, vv})
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].V > arr[j].V })
	if len(arr) > k {
		arr = arr[:k]
	}
	out := []string{}
	for _, it := range arr {
		out = append(out, fmt.Sprintf("  %-25s  %.3f", it.K, it.V))
	}
	return out
}

func doCheckFixedReplay(g Gains, f Floors, mem *Memory) {
	fmt.Println("\nDO-CHECK (planning-critical, VALID replay) [DETERMINISTIC]: TRUE risk=0.72, calm (uncertain obs -> WAIT should help)")
	ctx := Ctx{
		TrueRisk: 0.72,
		Emotion:  CALM,
		Obs1:     0.58,
		Rel1:     0.40,
		Obs2:     0.74,
		Rel2:     0.92,
	}

	baseRng := rand.New(rand.NewSource(7))
	mode := EvalGreedy()

	runEpisode := func(name string, tog Toggle) {
		fmt.Printf("\n%s episode:\n", name)
		s1 := step1Policy(ctx, g, f, mem, tog, mode, baseRng)
		fmt.Printf("  step1=%s dist1={BUY:%.3f, WAIT:%.3f, RUN:%.3f}  VOI=%.3f waitLogit=%.3f  trace1=%s\n",
			s1.Action.String(), s1.Dist[BUY], s1.Dist[WAIT], s1.Dist[RUN], s1.VOI, s1.WaitLogit, s1.Trace)

		usedWait := false
		final := s1.Action
		var s2 StepResult
		if s1.Action == WAIT {
			usedWait = true
			s2 = step2Policy(ctx, g, f, mem, tog, mode, baseRng)
			fmt.Printf("  step2=%s dist2={BUY:%.3f, RUN:%.3f} trace2=%s\n",
				s2.Action.String(), s2.Dist[BUY], s2.Dist[RUN], s2.Trace)
			final = s2.Action
		}

		r, cat := reward(ctx.TrueRisk, final, usedWait)
		fmt.Printf("  final=%s reward=%.2f cat=%v (fixed obs1=0.58 rel1=0.40 | obs2=0.74 rel2=0.92)\n",
			final.String(), r, cat)

		keys := []string{"Goal", "Will", "Instinct", "Logic", "Emotion", "Memory"}
		fmt.Printf("  contrib(step1 chosen): ")
		first := true
		for _, k := range keys {
			if v, ok := s1.Contrib[k]; ok && math.Abs(v) > 1e-9 {
				if !first {
					fmt.Printf("  ")
				}
				fmt.Printf("%s=%.3f", k, v)
				first = false
			}
		}
		if first {
			fmt.Printf("(none)")
		}
		fmt.Println()
	}

	runEpisode("BASELINE", Toggle{Goal: true, Emotion: true, Memory: true, Inst: true, Logic: true, Will: true})
	runEpisode("do(DISABLE Logic)", Toggle{Goal: true, Emotion: true, Memory: true, Inst: true, Logic: false, Will: true})
	runEpisode("do(DISABLE Memory)", Toggle{Goal: true, Emotion: true, Memory: false, Inst: true, Logic: true, Will: true})
	runEpisode("do(DISABLE Instinct)", Toggle{Goal: true, Emotion: true, Memory: true, Inst: false, Logic: true, Will: true})
}

func main() {
	fmt.Println("=== STB TIME+PLANNING DEMO v6.5.7 (Go) ===")
	fmt.Println("Fixes:")
	fmt.Println("  (1) do-check + evaluation = deterministic argmax (no sampling / no exploration)")
	fmt.Println("  (2) Memory AVOID-ONLY w/ cap + decay (brain-like forgetting)")
	fmt.Println("  (3) Scientific polish: show VOI + waitLogit; Memory penalizes BUY only in MED/HIGH (LOW=0)")
	fmt.Println("  (4) Greedy planning stats separated from training noise")
	fmt.Println("\nNew: 2-step planning via WAIT (observe) -> improved estimate -> final BUY/RUN.")
	fmt.Println("Goal: Logic becomes causally necessary for choosing WAIT under uncertainty.\n")

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// init
	g := Gains{Goal: 1.0, Emotion: 1.0, Memory: 1.0, Inst: 1.0, Logic: 1.0, Will: 1.0}
	f := Floors{Emotion: 0.70, Memory: 0.70, Inst: 0.70, Logic: 0.70, Will: 0.70}

	// Memory controls (tuned for demo):
	// cap prevents endless trauma growth; decay forgets slowly if not reinforced.
	memCap := 8.0
	memDecay := 0.995
	mem := NewMemory(memCap, memDecay)

	fmt.Println("BEFORE TRAINING (GREEDY policy stats):")
	rows := []StatsRow{
		policyStatsGreedy(rng, g, f, mem, MED, CALM, 300),
		policyStatsGreedy(rng, g, f, mem, MED, FEAR, 300),
		policyStatsGreedy(rng, g, f, mem, HIGH, CALM, 300),
		policyStatsGreedy(rng, g, f, mem, HIGH, FEAR, 300),
		policyStatsGreedy(rng, g, f, mem, LOW, CALM, 300),
	}
	for _, r := range rows {
		fmt.Printf("  %-26s | WAIT%%=%3d  Final(BUY/RUN)=%3d/%3d  avgR=%.3f\n",
			r.Label, r.WaitPct, r.BuyPct, r.RunPct, r.AvgR)
	}

	fmt.Println("\nLEARNED GAINS (init):")
	fmt.Printf("  Goal     %.3f\n  Emotion  %.3f\n  Memory   %.3f\n  Instinct %.3f\n  Logic    %.3f\n  Will     %.3f\n\n",
		g.Goal, g.Emotion, g.Memory, g.Inst, g.Logic, g.Will)

	episodes := 1200
	fmt.Printf("TRAINING (%d episodes)\nEvery 200 eps: show avg reward + gains + top memory.\n\n", episodes)

	roll := 0.0
	for ep := 1; ep <= episodes; ep++ {
		r := trainEpisode(rng, &g, f, mem)
		roll = 0.95*roll + 0.05*r

		if ep%200 == 0 {
			fmt.Printf("ep=%d  rollR=%.3f\n", ep, roll)
			fmt.Println("LEARNED GAINS:")
			fmt.Printf("  Goal     %.3f\n  Emotion  %.3f\n  Memory   %.3f\n  Instinct %.3f\n  Logic    %.3f\n  Will     %.3f\n",
				g.Goal, g.Emotion, g.Memory, g.Inst, g.Logic, g.Will)
			fmt.Println("\nMEMORY TOP (cap+decay active):")
			for _, line := range topMemory(mem, 8) {
				fmt.Println(line)
			}
			fmt.Println()
		}
	}

	fmt.Println("\nAFTER TRAINING (GREEDY policy stats):")
	rows = []StatsRow{
		policyStatsGreedy(rng, g, f, mem, MED, CALM, 400),
		policyStatsGreedy(rng, g, f, mem, MED, FEAR, 400),
		policyStatsGreedy(rng, g, f, mem, HIGH, CALM, 400),
		policyStatsGreedy(rng, g, f, mem, HIGH, FEAR, 400),
		policyStatsGreedy(rng, g, f, mem, LOW, CALM, 400),
	}
	for _, r := range rows {
		fmt.Printf("  %-26s | WAIT%%=%3d  Final(BUY/RUN)=%3d/%3d  avgR=%.3f\n",
			r.Label, r.WaitPct, r.BuyPct, r.RunPct, r.AvgR)
	}

	fmt.Println("\nLEARNED GAINS:")
	fmt.Printf("  Goal     %.3f\n  Emotion  %.3f\n  Memory   %.3f\n  Instinct %.3f\n  Logic    %.3f\n  Will     %.3f\n",
		g.Goal, g.Emotion, g.Memory, g.Inst, g.Logic, g.Will)

	fmt.Println("\nMEMORY TOP:")
	for _, line := range topMemory(mem, 10) {
		fmt.Println(line)
	}

	doCheckFixedReplay(g, f, mem)

	fmt.Println("\nDone.")
}
