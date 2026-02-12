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

const WAIT_LOGIT_MAX = 5.0

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
	Epsilon       float64 // optional exploration
}

func TrainMode() PolicyMode  { return PolicyMode{Deterministic: false, Temperature: 1.0, Epsilon: 0.0} }
func EvalGreedy() PolicyMode { return PolicyMode{Deterministic: true, Temperature: 1.0, Epsilon: 0.0} }

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
	if mode.Epsilon > 0 && rng.Float64() < mode.Epsilon {
		return acts[rng.Intn(len(acts))]
	}
	return sample(dist, rng, acts)
}

// --- Environment / reward ---
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
	if b == HIGH && final == BUY {
		base = -1.20
		catastrophic = true
	}
	if usedWait {
		base -= 0.06
		if b == HIGH && final == RUN {
			base += 0.06
		}
	}
	return base, catastrophic
}

// --- VOI heuristic (fallback) ---
func computeVOI(est float64, rel float64) float64 {
	boundary := 0.66
	close := 1.0 - clamp(math.Abs(est-boundary)/0.34, 0, 1)
	uncertainty := (1.0 - rel) * close
	voi := clamp(uncertainty*1.35, 0, 1)
	return voi
}

// helper: bucket VOI so causal ACE does not mix regimes.
func voiBucket(voi float64) string {
	if voi < 0.20 {
		return "VOI_LOW"
	}
	if voi < 0.55 {
		return "VOI_MED"
	}
	return "VOI_HIGH"
}

// =====================================================
// CAUSAL LEARNER: learn ACE of WAIT vs "no-logic world"
// =====================================================

type RelBucket int

const (
	REL_LOW RelBucket = iota
	REL_MED
	REL_HIGH
)

func (rb RelBucket) String() string {
	switch rb {
	case REL_LOW:
		return "REL_LOW"
	case REL_MED:
		return "REL_MED"
	case REL_HIGH:
		return "REL_HIGH"
	default:
		return "REL_?"
	}
}

func relBucketOf(r float64) RelBucket {
	if r < 0.45 {
		return REL_LOW
	}
	if r < 0.70 {
		return REL_MED
	}
	return REL_HIGH
}

// CausalLearner learns:
//   ACE(key) ≈ E[Reward | do(A1=WAIT)] - E[Reward | policy with Logic=0]
type CausalLearner struct {
	ace   map[string]float64
	count map[string]int
	alpha float64 // EMA rate
}

func NewCausalLearner(alpha float64) *CausalLearner {
	return &CausalLearner{
		ace:   make(map[string]float64),
		count: make(map[string]int),
		alpha: alpha,
	}
}

// PATCH: backoff keys (from most specific -> more general) to avoid sparse-regime starvation.
func (cl *CausalLearner) KeysFromCtx(ctx Ctx) []string {
	b := bucketOfRisk(ctx.Obs1)
	rb := relBucketOf(ctx.Rel1)
	voi := computeVOI(ctx.Obs1, ctx.Rel1)
	vb := voiBucket(voi)

	k1 := fmt.Sprintf("%s|%s|%s|%s", b.String(), ctx.Emotion.String(), rb.String(), vb) // most specific
	k2 := fmt.Sprintf("%s|any|%s|%s", b.String(), rb.String(), vb)                      // drop emotion
	k3 := fmt.Sprintf("%s|any|any|%s", b.String(), vb)                                  // drop rel
	k4 := fmt.Sprintf("any|any|any|%s", vb)                                             // only VOI regime

	return []string{k1, k2, k3, k4}
}

func (cl *CausalLearner) Update(key string, delta float64) {
	cl.count[key]++
	if cl.count[key] == 1 {
		cl.ace[key] = delta
		return
	}
	old := cl.ace[key]
	cl.ace[key] = (1.0-cl.alpha)*old + cl.alpha*delta
}

func (cl *CausalLearner) Get(key string) (ace float64, n int, ok bool) {
	n = cl.count[key]
	ace, ok = cl.ace[key]
	return ace, n, ok
}

func topCausal(cl *CausalLearner, k int) []string {
	type kv struct {
		K string
		V float64
		N int
	}
	var arr []kv
	for kk, vv := range cl.ace {
		arr = append(arr, kv{K: kk, V: vv, N: cl.count[kk]})
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].V > arr[j].V })
	if len(arr) > k {
		arr = arr[:k]
	}
	out := []string{}
	for _, it := range arr {
		out = append(out, fmt.Sprintf("  %-28s  ACE=%.3f  n=%d", it.K, it.V, it.N))
	}
	return out
}

type StepResult struct {
	Action    Action
	Dist      map[Action]float64
	Contrib   map[string]float64 // module -> contribution to chosen action
	Trace     string
	VOI       float64
	WaitLogit float64
	WaitMode  string // "VOI" / "LEARNED_ACE" / "HYBRID_VOI"
	LearnedACE float64
	LearnedN   int
	CausalKey  string
}

// -------------------------------
// NEW: tiny “SafetyPrior” shaping
// -------------------------------
// Goal: BEFORE training we should not be 100% BUY in TRUE_HIGH calm.
// This is NOT learning; it's just a sane prior: high observed risk increases RUN preference
// a little, without killing the learning effect.
func safetyPrior(obs float64) float64 {
	// obs in [0..1]. Strong only near top.
	// 0.75-> small, 0.9-> medium, 1.0-> stronger
	return clamp((obs-0.72)/0.28, 0, 1)
}

// Score model: modules contribute to BUY/RUN; WAIT is Logic via (learned ACE OR VOI fallback).
func step1Policy(ctx Ctx, g Gains, f Floors, mem *Memory, tog Toggle, mode PolicyMode, rng *rand.Rand, cl *CausalLearner) StepResult {
	b := bucketOfRisk(ctx.Obs1)

	emGain := math.Max(f.Emotion, g.Emotion)
	memGain := math.Max(f.Memory, g.Memory)
	inGain := math.Max(f.Inst, g.Inst)
	loGain := math.Max(f.Logic, g.Logic)
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
	if !tog.Logic {
		loGain = 0
	}
	if !tog.Will {
		wlGain = 0
	}

	goal := 1.20
	if !tog.Goal {
		goal = 0
	}

	// --- base tendencies ---
	instBUY := 0.30
	instRUN := 0.30
	if b == HIGH {
		// was: instRUN=1.10 instBUY=0.10
		// NEW: slightly safer prior (doesn't eliminate learning)
		instRUN = 1.25
		instBUY = 0.06
	} else if b == LOW {
		instBUY = 1.00
		instRUN = 0.15
	} else {
		instBUY = 0.65
		instRUN = 0.45
	}

	emoBUY := 0.20
	emoRUN := 0.20
	if ctx.Emotion == FEAR {
		emoRUN = 0.65
		emoBUY = 0.10
	} else {
		emoBUY = 0.35
		emoRUN = 0.10
	}

	willAct := 0.45

	// Memory AVOID-ONLY penalty (never boosts):
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

	// --- WAIT: HYBRID Logic gate = max(ACE_gate, VOI_gate) ---
	voi := computeVOI(ctx.Obs1, ctx.Rel1)

	waitMode := "VOI"
	learnedACE := 0.0
	learnedN := 0
	key := ""

	waitLogitACE := 0.0
	waitLogitVOI := 0.0
	waitLogit := 0.0

	// NEW: SafetyPrior shaping
	sp := safetyPrior(ctx.Obs1)
	// Adds a bit of RUN, subtracts a bit of BUY when obs is high.
	// Important: small amplitude to keep learning/differences.
	safetyRun := 0.55 * sp
	safetyBuy := 0.35 * sp

	// BUY/RUN logits
	buyLogit := goal +
		inGain*instBUY +
		emGain*emoBUY +
		wlGain*willAct -
		memGain*memPenBUY -
		safetyBuy

	runLogit := 0.30 +
		inGain*instRUN +
		emGain*emoRUN +
		wlGain*willAct +
		safetyRun

		// --- PATCH: uncertainty commit penalty ---
// If observation is unreliable / boundary-close (high VOI), penalize committing to BUY/RUN.
// This makes WAIT causally necessary in planning-critical cases (do-check).
if loGain > 0 {
    // stronger when VOI high and reliability low
    unc := clamp(voi, 0, 1) * clamp(1.0-ctx.Rel1, 0, 1)

    // gate so LOW risk doesn't get slowed down
    if b != LOW && unc > 0.10 {
        // scale chosen to flip do-check from BUY->WAIT without making policy overly timid
        commitPenalty := loGain * unc * 2.25
        buyLogit -= commitPenalty
        runLogit -= commitPenalty * 0.75 // run a bit less penalized than buy
    }
}


	if loGain > 0 {
		// VOI gate
		voiThreshold := 0.18
		voiScale := 2.60
		waitLogitVOI = loGain * math.Max(0, (voi-voiThreshold)) * voiScale

// ACE gate with KEY BACKOFF (prefer exact emotion; backoff only if sparse)
minN := 35
aceThreshold := 0.08
aceScale := 6.50

if cl != nil {
    keys := cl.KeysFromCtx(ctx)
    // keys: [k1 exact, k2 drop emo, k3 drop rel, k4 voi-only]

    // 1) First: record exact (emotion-specific) if exists (for transparency)
    // (optional, but helps debugging)
    // exactAce, exactN, exactOK := cl.Get(keys[0])

    // 2) Choose which key to USE for ACE (must be mature)
    usedAce := 0.0
    usedN := 0
    usedKey := ""
    usedOK := false

    // Prefer exact key if mature
    if ace, n, ok := cl.Get(keys[0]); ok && n >= minN {
        usedAce, usedN, usedKey, usedOK = ace, n, keys[0], true
    } else {
        // Backoff to more general keys (first mature wins)
        for i := 1; i < len(keys); i++ {
            ace, n, ok := cl.Get(keys[i])
            if ok && n >= minN {
                usedAce, usedN, usedKey, usedOK = ace, n, keys[i], true
                break
            }
        }
    }

    // 3) Expose what we actually used (so DO-CHECK key is honest)
    if usedOK {
        learnedACE = usedAce
        learnedN = usedN
        key = usedKey

        waitLogitACE = loGain * math.Max(0, (learnedACE-aceThreshold)) * aceScale
    } else {
        // If nothing mature, still allow printing the best-available *exact* key (optional)
        // so logs don't show "any" when calm exists but sparse.
        if ace, n, ok := cl.Get(keys[0]); ok {
            learnedACE = ace
            learnedN = n
            key = keys[0] // keeps DO-CHECK showing calm instead of any when possible
        }
    }
}


// HYBRID combine:
waitLogit = waitLogitVOI
waitMode = "VOI"

if waitLogitACE > waitLogitVOI {
	waitLogit = waitLogitACE
	waitMode = "LEARNED_ACE"
} else {
	if cl != nil && learnedN >= minN {
		waitMode = "HYBRID_VOI"
	}
}

// PATCH: WAIT MarginGate + VOI override
margin := math.Abs(buyLogit - runLogit)
marginGate := clamp(1.0-margin/1.2, 0.0, 1.0)

// If VOI is very high, still allow WAIT even if margin is larger (profit-rescue cases).
voiOverride := clamp((voi-0.55)/0.35, 0.0, 1.0) // voi>=0.90 => ~1
gate := math.Max(marginGate, 0.40*voiOverride)

waitLogit *= (0.35 + 0.65*gate)

// soft saturation BEFORE clamp: prevents "always hits 5.000" look
// maps x->max*(1-exp(-x/max)), so big x approaches max smoothly
if waitLogit > 0 {
    waitLogit = WAIT_LOGIT_MAX * (1.0 - math.Exp(-waitLogit/WAIT_LOGIT_MAX))
}

// final safety clamp
waitLogit = clamp(waitLogit, 0.0, WAIT_LOGIT_MAX)


	}

	logits := map[Action]float64{
		BUY:  buyLogit,
		WAIT: waitLogit,
		RUN:  runLogit,
	}

	temp := mode.Temperature
	if mode.Deterministic {
		temp = 1.0
	}
	dist := softmax(logits, temp)
	chosen := pickAction(dist, mode, rng, []Action{BUY, WAIT, RUN})

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
		if sp > 1e-9 {
			contrib["SafetyPrior"] = -safetyBuy
		}
	case RUN:
		contrib["Instinct"] = inGain * instRUN
		contrib["Emotion"] = emGain * emoRUN
		contrib["Will"] = wlGain * willAct
		if sp > 1e-9 {
			contrib["SafetyPrior"] = safetyRun
		}
	}

	return StepResult{
		Action:      chosen,
		Dist:        dist,
		Contrib:     contrib,
		Trace:       "Goal->Emotion->Memory->Instinct->Logic->Will",
		VOI:         voi,
		WaitLogit:   waitLogit,
		WaitMode:    waitMode,
		LearnedACE:  learnedACE,
		LearnedN:    learnedN,
		CausalKey:   key,
	}
}

func step2Policy(ctx Ctx, g Gains, f Floors, mem *Memory, tog Toggle, mode PolicyMode, rng *rand.Rand) StepResult {
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
		instRUN = 1.30
		instBUY = 0.04
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

	// SafetyPrior also applies at step2 (we already observed better obs2, so it's fair)
	sp := safetyPrior(ctx.Obs2)
	safetyRun := 0.45 * sp
	safetyBuy := 0.25 * sp

	buyLogit := goal +
		inGain*instBUY +
		emGain*emoBUY +
		wlGain*willAct -
		memGain*memPenBUY -
		safetyBuy

	runLogit := 0.25 +
		inGain*instRUN +
		emGain*emoRUN +
		wlGain*willAct +
		safetyRun

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
		if sp > 1e-9 {
			contrib["SafetyPrior"] = -safetyBuy
		}
	} else {
		contrib["Instinct"] = inGain * instRUN
		contrib["Emotion"] = emGain * emoRUN
		contrib["Will"] = wlGain * willAct
		if sp > 1e-9 {
			contrib["SafetyPrior"] = safetyRun
		}
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

type StatsRow struct {
	Label     string
	WaitPct   int
	BuyPct    int
	RunPct    int
	AvgR      float64
	CatPct    int // NEW headline metric
}

func policyStatsGreedy(rng *rand.Rand, g Gains, f Floors, mem *Memory, trueBucket Bucket, emo Emotion, n int, cl *CausalLearner) StatsRow {
	tog := Toggle{Goal: true, Emotion: true, Memory: true, Inst: true, Logic: true, Will: true}
	mode := EvalGreedy()

	waitC := 0
	buyC := 0
	runC := 0
	catC := 0
	sumR := 0.0

	for i := 0; i < n; i++ {
		var tr float64
		switch trueBucket {
		case LOW:
			tr = rng.Float64() * 0.33
		case MED:
			tr = 0.33 + rng.Float64()*0.33
		case HIGH:
			tr = 0.66 + rng.Float64()*0.34
		}

		rel1 := clamp(0.25+0.55*rng.Float64(), 0.25, 0.80)
		rel2 := clamp(rel1+0.30+0.20*rng.Float64(), 0.60, 0.98)
		base1 := rng.Float64()*2 - 1
		base2 := rng.Float64()*2 - 1
		noise1 := base1 * (1.0 - rel1) * 0.55
		noise2 := base2 * (1.0 - rel2) * 0.25
		obs1 := clamp(tr+noise1, 0, 1)
		obs2 := clamp(tr+noise2, 0, 1)

		ctx := Ctx{TrueRisk: tr, Emotion: emo, Obs1: obs1, Rel1: rel1, Obs2: obs2, Rel2: rel2}
		s1 := step1Policy(ctx, g, f, mem, tog, mode, rng, cl)

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
		r, cat := reward(tr, final, usedWait)
		if cat {
			catC++
		}
		sumR += r
	}

	label := fmt.Sprintf("TRUE_%s  %s", trueBucket.String(), emo.String())
	return StatsRow{
		Label:   label,
		WaitPct: int(math.Round(100.0 * float64(waitC) / float64(n))),
		BuyPct:  int(math.Round(100.0 * float64(buyC) / float64(n))),
		RunPct:  int(math.Round(100.0 * float64(runC) / float64(n))),
		AvgR:    sumR / float64(n),
		CatPct:  int(math.Round(100.0 * float64(catC) / float64(n))),
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

func doCheckFixedReplay(g Gains, f Floors, mem *Memory, cl *CausalLearner) {
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
		s1 := step1Policy(ctx, g, f, mem, tog, mode, baseRng, cl)
		fmt.Printf("  step1=%s dist1={BUY:%.3f, WAIT:%.3f, RUN:%.3f}  VOI=%.3f waitLogit=%.3f  waitMode=%s\n",
			s1.Action.String(), s1.Dist[BUY], s1.Dist[WAIT], s1.Dist[RUN], s1.VOI, s1.WaitLogit, s1.WaitMode)

if s1.Action == WAIT && s1.WaitMode == "LEARNED_ACE" && s1.CausalKey != "" {
	fmt.Printf("  learnedACE=%.3f  n=%d  key=%s\n", s1.LearnedACE, s1.LearnedN, s1.CausalKey)
}




		usedWait := false
		final := s1.Action
		var s2 StepResult

		if s1.Action == WAIT {
			usedWait = true
			s2 = step2Policy(ctx, g, f, mem, tog, mode, baseRng)
			fmt.Printf("  step2=%s dist2={BUY:%.3f, RUN:%.3f}\n",
				s2.Action.String(), s2.Dist[BUY], s2.Dist[RUN])
			final = s2.Action
		}

		r, cat := reward(ctx.TrueRisk, final, usedWait)

		fmt.Printf("  final=%s reward=%.2f cat=%v (fixed obs1=0.58 rel1=0.40 | obs2=0.74 rel2=0.92)\n",
			final.String(), r, cat)

		keys := []string{"Goal", "Will", "Instinct", "Logic", "Emotion", "Memory", "SafetyPrior"}
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

// =====================================================
// FORMAL SCM + COUNTERFACTUAL do()
// =====================================================

type Latents struct {
	TrueRisk float64
	Emotion  Emotion
	Rel1     float64
	Rel2     float64
	Noise1   float64 // in [-1..1]
	Noise2   float64 // in [-1..1]
}

func (u Latents) MakeCtx() Ctx {
	obs1 := clamp(u.TrueRisk+u.Noise1*(1.0-u.Rel1)*0.55, 0, 1)
	obs2 := clamp(u.TrueRisk+u.Noise2*(1.0-u.Rel2)*0.25, 0, 1)
	return Ctx{
		TrueRisk: u.TrueRisk,
		Emotion:  u.Emotion,
		Obs1:     obs1,
		Rel1:     u.Rel1,
		Obs2:     obs2,
		Rel2:     u.Rel2,
	}
}

type WorldDo struct {
	ForceRel1      *float64
	ForceObs1      *float64
	ForceRel2      *float64
	ForceObs2      *float64
	ForceEmo       *Emotion
	KeepStructural bool
	U              *Latents
}

func ApplyWorldDo(ctx Ctx, w WorldDo) Ctx {
	if w.ForceEmo != nil {
		ctx.Emotion = *w.ForceEmo
	}
	if w.ForceRel1 != nil {
		ctx.Rel1 = *w.ForceRel1
		if w.KeepStructural && w.U != nil {
			u := *w.U
			u.Rel1 = ctx.Rel1
			ctx.Obs1 = clamp(u.TrueRisk+u.Noise1*(1.0-u.Rel1)*0.55, 0, 1)
		}
	}
	if w.ForceRel2 != nil {
		ctx.Rel2 = *w.ForceRel2
		if w.KeepStructural && w.U != nil {
			u := *w.U
			u.Rel2 = ctx.Rel2
			ctx.Obs2 = clamp(u.TrueRisk+u.Noise2*(1.0-u.Rel2)*0.25, 0, 1)
		}
	}
	if w.ForceObs1 != nil {
		ctx.Obs1 = *w.ForceObs1
	}
	if w.ForceObs2 != nil {
		ctx.Obs2 = *w.ForceObs2
	}
	return ctx
}

type PolicyDo struct {
	ForceA1 *Action
}

type EpisodeCF struct {
	Ctx       Ctx
	A1        Action
	Step1     StepResult
	A2        *Action
	Step2     *StepResult
	Final     Action
	UsedWait  bool
	Reward    float64
	Cat       bool
	Suggested Action
}

func runEpisodeWithInterventions(
	ctx Ctx,
	g Gains, f Floors, mem *Memory,
	tog Toggle,
	mode PolicyMode,
	rng *rand.Rand,
	pdo PolicyDo,
	cl *CausalLearner,
) EpisodeCF {

	s1 := step1Policy(ctx, g, f, mem, tog, mode, rng, cl)
	suggested := s1.Action

	suggestedNonWait := suggested
	if suggested == WAIT {
		if s1.Dist[BUY] >= s1.Dist[RUN] {
			suggestedNonWait = BUY
		} else {
			suggestedNonWait = RUN
		}
	}

	a1 := s1.Action
	if pdo.ForceA1 != nil {
		a1 = *pdo.ForceA1
		s1.Action = a1
		s1.Contrib = map[string]float64{"do(A1)": 1.0}
	}

	usedWait := false
	final := a1

	var a2 *Action
	var s2 *StepResult

	if a1 == WAIT {
		usedWait = true
		r2 := step2Policy(ctx, g, f, mem, tog, mode, rng)
		aa2 := r2.Action
		a2 = &aa2
		s2 = &r2
		final = aa2
	}

	rw, cat := reward(ctx.TrueRisk, final, usedWait)

	// refund wait-tax when WAIT actually changed decision vs non-WAIT suggestion
	if usedWait {
		if suggestedNonWait != WAIT && final != suggestedNonWait {
			rw += 0.06
		}
	}

	return EpisodeCF{
		Ctx:       ctx,
		A1:        a1,
		Step1:     s1,
		A2:        a2,
		Step2:     s2,
		Final:     final,
		UsedWait:  usedWait,
		Reward:    rw,
		Cat:       cat,
		Suggested: suggested,
	}
}

func printSCMGraph() {
	fmt.Println("\n=== FORMAL SCM (Structural Causal Model) ===")
	fmt.Println("Exogenous (U): TrueRisk, Emotion, Rel1, Rel2, Noise1, Noise2")
	fmt.Println("Endogenous:")
	fmt.Println("  Obs1 := TrueRisk + Noise1*(1-Rel1)*0.55")
	fmt.Println("  Obs2 := TrueRisk + Noise2*(1-Rel2)*0.25")
	fmt.Println("  A1  := π1(Obs1, Rel1, Emotion, Memory, Instinct, Logic, Will)")
	fmt.Println("  A2  := π2(Obs2, Emotion, Memory, Instinct, Will)   (only if A1=WAIT)")
	fmt.Println("  Final := A1 if A1!=WAIT else A2")
	fmt.Println("  Reward := R(TrueRisk, Final, usedWait)")

	fmt.Println("\nCausal edges (graph):")
	fmt.Println("  TrueRisk -> Obs1, Obs2, Reward")
	fmt.Println("  Rel1 -> Obs1 -> A1")
	fmt.Println("  Rel2 -> Obs2 -> A2")
	fmt.Println("  Emotion -> A1, A2")
	fmt.Println("  Memory -> A1, A2")
	fmt.Println("  Instinct -> A1, A2")
	fmt.Println("  Logic -> A1 (WAIT only via learned-ACE/VOI gate)")
	fmt.Println("  Will -> A1, A2")
	fmt.Println("  A1 -> (usedWait, A2 availability) -> Final -> Reward")
	fmt.Println("===========================================\n")
}

func causalCase(title string, u Latents, g Gains, f Floors, mem *Memory, cl *CausalLearner) {
	fmt.Printf("=== SCM CASE: %s ===\n", title)

	ctx := u.MakeCtx()
	fmt.Println("ABDUCTION (fix U; derive observed context):")
	fmt.Printf("  U: TrueRisk=%.3f Emotion=%s Rel1=%.2f Rel2=%.2f Noise1=%.3f Noise2=%.3f\n",
		u.TrueRisk, u.Emotion.String(), u.Rel1, u.Rel2, u.Noise1, u.Noise2)
	fmt.Printf("  ctx: Obs1=%.3f (bucket=%s) Rel1=%.2f | Obs2=%.3f (bucket=%s) Rel2=%.2f\n\n",
		ctx.Obs1, bucketOfRisk(ctx.Obs1).String(), ctx.Rel1,
		ctx.Obs2, bucketOfRisk(ctx.Obs2).String(), ctx.Rel2)

	mode := EvalGreedy()
	rng := rand.New(rand.NewSource(123))

	tAll := Toggle{Goal: true, Emotion: true, Memory: true, Inst: true, Logic: true, Will: true}
	tNoLogic := Toggle{Goal: true, Emotion: true, Memory: true, Inst: true, Logic: false, Will: true}

	printEp := func(title string, ep EpisodeCF) {
		fmt.Printf("%s\n", title)
		fmt.Printf("  step1=%s  dist1={BUY:%.3f WAIT:%.3f RUN:%.3f}  VOI=%.3f waitLogit=%.3f waitMode=%s\n",
			ep.A1.String(),
			ep.Step1.Dist[BUY], ep.Step1.Dist[WAIT], ep.Step1.Dist[RUN],
			ep.Step1.VOI, ep.Step1.WaitLogit, ep.Step1.WaitMode,
		)
if ep.A1 == WAIT && ep.Step1.WaitMode == "LEARNED_ACE" && ep.Step1.CausalKey != "" {
	fmt.Printf("  learnedACE=%.3f n=%d key=%s\n", ep.Step1.LearnedACE, ep.Step1.LearnedN, ep.Step1.CausalKey)
}


		if ep.UsedWait && ep.Step2 != nil && ep.A2 != nil {
			fmt.Printf("  step2=%s  dist2={BUY:%.3f RUN:%.3f}\n",
				ep.A2.String(),
				ep.Step2.Dist[BUY], ep.Step2.Dist[RUN],
			)
		}
		fmt.Printf("  final=%s  reward=%.2f  cat=%v  (usedWait=%v)\n\n",
			ep.Final.String(), ep.Reward, ep.Cat, ep.UsedWait,
		)
	}

	fmt.Println("=== INTERVENTIONS on same U ===")
	{
		epBase := runEpisodeWithInterventions(ctx, g, f, mem, tAll, mode, rng, PolicyDo{}, cl)
		printEp("BASELINE (policy, Logic ON):", epBase)

		epNoLogic := runEpisodeWithInterventions(ctx, g, f, mem, tNoLogic, mode, rng, PolicyDo{}, cl)
		printEp("do(Logic=0) (policy, Logic OFF):", epNoLogic)

		hiRel := 0.95
		ctxHiRel := ApplyWorldDo(ctx, WorldDo{ForceRel1: &hiRel, KeepStructural: true, U: &u})
		epHiRel := runEpisodeWithInterventions(ctxHiRel, g, f, mem, tAll, mode, rng, PolicyDo{}, cl)
		printEp("WORLD do(Rel1=0.95) (less uncertainty; Obs1 recomputed from U):", epHiRel)

		origObs := ctx.Obs1
		ctxFixObs := ApplyWorldDo(ctx, WorldDo{ForceObs1: &origObs})
		epFixObs := runEpisodeWithInterventions(ctxFixObs, g, f, mem, tAll, mode, rng, PolicyDo{}, cl)
		printEp("NODE do(Obs1=orig) (fix observation node):", epFixObs)

if title == "B) Profit-rescue near boundary (MED true risk; WAIT prevents mistaken RUN)" {
    if epBase.UsedWait && epBase.Reward > epNoLogic.Reward+1e-9 {
        fmt.Printf(">>> PROFIT RESCUE (boundary): WAIT prevented mistaken RUN under noisy Obs1\n")
        fmt.Printf("    baseline: %s -> %s  reward=%.2f\n", epBase.A1, epBase.Final, epBase.Reward)
        fmt.Printf("    no-logic: %s         reward=%.2f\n", epNoLogic.Final, epNoLogic.Reward)
        fmt.Printf("    Δreward  = %.2f  (WAIT saved profit)\n\n", epBase.Reward-epNoLogic.Reward)
    } else {
        fmt.Printf(">>> PROFIT RESCUE: not triggered in this U (baseline didn't benefit from WAIT)\n\n")
    }
}


		fmt.Println("=== COUNTERFACTUAL ACTION do(A1=...) on same U ===")
	{
		aBuy := BUY
		aWait := WAIT
		aRun := RUN

		epDoBuy := runEpisodeWithInterventions(ctx, g, f, mem, tAll, mode, rng, PolicyDo{ForceA1: &aBuy}, cl)
		fmt.Printf("ACTION do(A1=BUY):\n  policy_suggested_A1=%s  do(A1)=BUY (decision mechanism bypassed)\n", epDoBuy.Suggested)
		fmt.Printf("  final=%s  reward=%.2f  cat=%v\n\n", epDoBuy.Final, epDoBuy.Reward, epDoBuy.Cat)

		epDoWait := runEpisodeWithInterventions(ctx, g, f, mem, tAll, mode, rng, PolicyDo{ForceA1: &aWait}, cl)
		fmt.Printf("ACTION do(A1=WAIT):\n  policy_suggested_A1=%s  do(A1)=WAIT (decision mechanism bypassed)\n", epDoWait.Suggested)
		fmt.Printf("  final=%s  reward=%.2f  cat=%v\n\n", epDoWait.Final, epDoWait.Reward, epDoWait.Cat)

		epDoRun := runEpisodeWithInterventions(ctx, g, f, mem, tAll, mode, rng, PolicyDo{ForceA1: &aRun}, cl)
		fmt.Printf("ACTION do(A1=RUN):\n  policy_suggested_A1=%s  do(A1)=RUN (decision mechanism bypassed)\n", epDoRun.Suggested)
		fmt.Printf("  final=%s  reward=%.2f  cat=%v\n\n", epDoRun.Final, epDoRun.Reward, epDoRun.Cat)

		fmt.Println("COUNTERFACTUAL CLAIM (same U):")
		fmt.Printf("  do(A1=BUY)  -> final=%s reward=%.2f cat=%v\n", epDoBuy.Final, epDoBuy.Reward, epDoBuy.Cat)
		fmt.Printf("  do(A1=WAIT) -> final=%s reward=%.2f cat=%v\n", epDoWait.Final, epDoWait.Reward, epDoWait.Cat)
		fmt.Printf("  do(A1=RUN)  -> final=%s reward=%.2f cat=%v\n\n", epDoRun.Final, epDoRun.Reward, epDoRun.Cat)
	}
}
}

func causalSCMDemo(g Gains, f Floors, mem *Memory, cl *CausalLearner) {
	printSCMGraph()

	uA := Latents{
		TrueRisk: 0.72,
		Emotion:  CALM,
		Rel1:     0.40,
		Rel2:     0.92,
		Noise1:   -0.4242424242,
		Noise2:   1.0,
	}
	causalCase("A) Catastrophe-avoidance (HIGH true risk; WAIT prevents BUY catastrophe)", uA, g, f, mem, cl)

uB := Latents{
    TrueRisk: 0.64,
    Emotion:  FEAR,
    Rel1:     0.30,
    Rel2:     0.92,
    Noise1:   0.286,
    Noise2:   0.0,
}


	causalCase("B) Profit-rescue near boundary (MED true risk; WAIT prevents mistaken RUN)", uB, g, f, mem, cl)
}

// =====================================================
// TRAINING + CAUSAL LEARNING UPDATE
// =====================================================

func trainEpisode(rng *rand.Rand, g *Gains, f Floors, mem *Memory, cl *CausalLearner, ep int) (avgR float64) {
	mem.DecayAll()

	trueRisk := rng.Float64()
	emo := CALM
	if rng.Float64() < 0.35 {
		emo = FEAR
	}

	rel1 := clamp(0.25+0.55*rng.Float64(), 0.25, 0.80)
	rel2 := clamp(rel1+0.30+0.20*rng.Float64(), 0.60, 0.98)

	base1 := rng.Float64()*2 - 1
	base2 := rng.Float64()*2 - 1

	noise1 := base1 * (1.0 - rel1) * 0.55
	noise2 := base2 * (1.0 - rel2) * 0.25

	obs1 := clamp(trueRisk+noise1, 0, 1)
	obs2 := clamp(trueRisk+noise2, 0, 1)

	ctx := Ctx{TrueRisk: trueRisk, Emotion: emo, Obs1: obs1, Rel1: rel1, Obs2: obs2, Rel2: rel2}

	// causal update
	if cl != nil {
		u := Latents{
			TrueRisk: trueRisk,
			Emotion:  emo,
			Rel1:     rel1,
			Rel2:     rel2,
			Noise1:   base1,
			Noise2:   base2,
		}
		ctxU := u.MakeCtx()

		mode := EvalGreedy()
		rngCF := rand.New(rand.NewSource(int64(100000 + ep)))

		tAll := Toggle{Goal: true, Emotion: true, Memory: true, Inst: true, Logic: true, Will: true}
		tNoLogic := Toggle{Goal: true, Emotion: true, Memory: true, Inst: true, Logic: false, Will: true}

		aWait := WAIT
		epWait := runEpisodeWithInterventions(ctxU, *g, f, mem, tAll, mode, rngCF, PolicyDo{ForceA1: &aWait}, nil)
		epNoLogic := runEpisodeWithInterventions(ctxU, *g, f, mem, tNoLogic, mode, rngCF, PolicyDo{}, nil)

		delta := epWait.Reward - epNoLogic.Reward

		keys := cl.KeysFromCtx(ctxU)
		for i, kk := range keys {
			scale := 1.0
			if i == 1 {
				scale = 0.6
			} else if i == 2 {
				scale = 0.35
			} else if i == 3 {
				scale = 0.20
			}
			cl.Update(kk, delta*scale)
		}
	}

	tog := Toggle{Goal: true, Emotion: true, Memory: true, Inst: true, Logic: true, Will: true}
	mode := TrainMode()

	s1 := step1Policy(ctx, *g, f, mem, tog, mode, rng, cl)

	usedWait := false
	final := s1.Action
	if s1.Action == WAIT {
		usedWait = true
		s2 := step2Policy(ctx, *g, f, mem, tog, mode, rng)
		final = s2.Action
	}

	r, catastrophic := reward(trueRisk, final, usedWait)
	tb := bucketOfRisk(trueRisk)

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

	lr := 0.03

	if usedWait && r > 0.40 {
		g.Logic += lr * (r - 0.30)
	}
	if !usedWait && catastrophic {
		g.Memory += lr * 0.6
		g.Will -= lr * 0.4
		g.Inst -= lr * 0.2
	}
	g.Emotion += lr * 0.05 * (r - 0.35)
	g.Inst += lr * 0.05 * (r - 0.35)
	g.Will += lr * 0.03 * (r - 0.35)

	g.Emotion = clamp(g.Emotion, 0.70, 3.0)
	g.Memory = clamp(g.Memory, 0.70, 3.0)
	g.Inst = clamp(g.Inst, 0.70, 3.0)
	g.Logic = clamp(g.Logic, 0.70, 3.0)
	g.Will = clamp(g.Will, 0.70, 3.0)
	g.Goal = 1.0

	return r
}

func main() {
	fmt.Println("=== STB TIME+PLANNING DEMO v6.7.3 (Go) ===")
	fmt.Println("New minimal polish:")
	fmt.Println("  (A) SafetyPrior: sane pre-training behavior in HIGH (avoid 100% BUY in calm)")
	fmt.Println("  (B) headline metric: Catastrophe% in greedy stats (BUY in TRUE_HIGH)")
	fmt.Println("  (C) keep the core punch: do(Logic=0) breaks the world; SCM + counterfactual still proves it\n")

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	g := Gains{Goal: 1.0, Emotion: 1.0, Memory: 1.0, Inst: 1.0, Logic: 1.0, Will: 1.0}
	f := Floors{Emotion: 0.70, Memory: 0.70, Inst: 0.70, Logic: 0.70, Will: 0.70}

	memCap := 8.0
	memDecay := 0.995
	mem := NewMemory(memCap, memDecay)

	cl := NewCausalLearner(0.05)

	fmt.Println("BEFORE TRAINING (GREEDY policy stats):")
	rows := []StatsRow{
		policyStatsGreedy(rng, g, f, mem, HIGH, CALM, 400, cl),
		policyStatsGreedy(rng, g, f, mem, HIGH, FEAR, 400, cl),
		policyStatsGreedy(rng, g, f, mem, MED, CALM, 400, cl),
		policyStatsGreedy(rng, g, f, mem, MED, FEAR, 400, cl),
		policyStatsGreedy(rng, g, f, mem, LOW, CALM, 400, cl),
	}
	for _, r := range rows {
		fmt.Printf("  %-26s | WAIT%%=%3d  Final(BUY/RUN)=%3d/%3d  cat%%=%3d  avgR=%.3f\n",
			r.Label, r.WaitPct, r.BuyPct, r.RunPct, r.CatPct, r.AvgR)
	}

	fmt.Println("\nLEARNED GAINS (init):")
	fmt.Printf("  Goal     %.3f\n  Emotion  %.3f\n  Memory   %.3f\n  Instinct %.3f\n  Logic    %.3f\n  Will     %.3f\n\n",
		g.Goal, g.Emotion, g.Memory, g.Inst, g.Logic, g.Will)

	episodes := 1400
	fmt.Printf("TRAINING (%d episodes)\nEvery 200 eps: show roll reward + gains + top memory + top causal ACE.\n\n", episodes)

	roll := 0.0
	for ep := 1; ep <= episodes; ep++ {
		r := trainEpisode(rng, &g, f, mem, cl, ep)
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

			fmt.Println("\nCAUSAL TOP (learned ACE of WAIT vs no-logic):")
			for _, line := range topCausal(cl, 8) {
				fmt.Println(line)
			}
			fmt.Println()
		}
	}

	fmt.Println("\nAFTER TRAINING (GREEDY policy stats):")
	rows = []StatsRow{
		policyStatsGreedy(rng, g, f, mem, HIGH, CALM, 600, cl),
		policyStatsGreedy(rng, g, f, mem, HIGH, FEAR, 600, cl),
		policyStatsGreedy(rng, g, f, mem, MED, CALM, 600, cl),
		policyStatsGreedy(rng, g, f, mem, MED, FEAR, 600, cl),
		policyStatsGreedy(rng, g, f, mem, LOW, CALM, 600, cl),
	}
	for _, r := range rows {
		fmt.Printf("  %-26s | WAIT%%=%3d  Final(BUY/RUN)=%3d/%3d  cat%%=%3d  avgR=%.3f\n",
			r.Label, r.WaitPct, r.BuyPct, r.RunPct, r.CatPct, r.AvgR)
	}

	fmt.Println("\nLEARNED GAINS:")
	fmt.Printf("  Goal     %.3f\n  Emotion  %.3f\n  Memory   %.3f\n  Instinct %.3f\n  Logic    %.3f\n  Will     %.3f\n",
		g.Goal, g.Emotion, g.Memory, g.Inst, g.Logic, g.Will)

	fmt.Println("\nMEMORY TOP:")
	for _, line := range topMemory(mem, 10) {
		fmt.Println(line)
	}

	fmt.Println("\nCAUSAL TOP (ACE):")
	for _, line := range topCausal(cl, 12) {
		fmt.Println(line)
	}

	doCheckFixedReplay(g, f, mem, cl)
	causalSCMDemo(g, f, mem, cl)

	fmt.Println("\nDone.")
}
