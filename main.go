package main

import (
  // "math"
  "math/rand"
  "fmt"
  ui "github.com/gizak/termui/v3"
  "github.com/gizak/termui/v3/widgets"
  "time"
  "log"
  "github.com/chewxy/math32"
  "gonum.org/v1/gonum/blas/blas32"
)

// ++++++++++++++++++++++++++++
// CONSTANTS

const second int = 1000
const ms int = second / 1000


// ++++++++++++++++++++++++++++
// NEURONS

type Layer interface {
  eeg(float32)
  quant() int
  history() []float64
  feed([]float32)
  pulse() *[]bool
  G() *[]float32
  V() *[]float32
  I() *[]float32
  geti() []float32
  getv() []float32
}


// NEURONS
// +++++++
// IZ

type IZPrm struct {
  a     	float32
  b     	float32
  c 		  float32
  d 		  float32
  vt    	float32
}

type IZ struct {
  prm 		*IZPrm
  n 		  int
  v 		  []float32
  u 		  []float32
  cndr 		[]bool
  cndrc 	[]float64
  i 		  []float32
}

func (nrnm *IZ) quant() int{
  return nrnm.n
}

func (nrnm *IZ) history() []float64 {
  return nrnm.cndrc
}

func (nrnm *IZ) feed(in []float32) {
  nrnm.i = in
}

func (nrnm *IZ) pulse() *[]bool {
  return &nrnm.cndr
}

func (nrnm *IZ) I() *[]float32 {
  return &nrnm.i
}

func (nrnm *IZ) G() error {
  return nil
}

func (nrnm *IZ) V() *[]float32 {
  return &nrnm.v
}

func (nrnm *IZ) geti() (out []float32) {
  out = nrnm.i
  return
}

func (nrnm *IZ) getv() (out []float32) {
  out = nrnm.v
  return
}

func (nrnm *IZ) eeg(dt float32) {

  for i:=0; i<nrnm.n; i++ {
    nrnm.v[i] += 0.5 * dt * (0.04 * (nrnm.v[i] * nrnm.v[i]) + 5 * nrnm.v[i] + 140 - nrnm.u[i] + nrnm.i[i])
    nrnm.v[i] += 0.5 * dt * (0.04 * (nrnm.v[i] * nrnm.v[i]) + 5 * nrnm.v[i] + 140 - nrnm.u[i] + nrnm.i[i])
    nrnm.u[i] += dt * (nrnm.prm.a * (nrnm.prm.b * nrnm.v[i] - nrnm.u[i]))
  }

  for i:=0; i<nrnm.n; i++ {
    switch {
    case nrnm.v[i] > nrnm.prm.vt:
       nrnm.cndr[i] = true
       nrnm.v[i] = nrnm.prm.c
       nrnm.u[i] += nrnm.prm.d
       nrnm.cndrc[i] += 1
    default:
       nrnm.cndr[i] = false
    }
  }
}

func iz_init(n int, prm *IZPrm) *IZ {
  return &IZ{
    prm:  	prm,
    n:    	n,
    v:    	make([]float32, n),
    u:    	make([]float32, n),
    cndr: 	make([]bool, n),
    cndrc:	make([]float64, n),
    i:    	make([]float32, n),
  }
}


// NEURONS
// +++++++
// IF

type IFPrm struct {
  tm     	float32
  te     	float32
  ti 	 	  float32
  vt 	 	  float32
  vr     	float32
  el 	 	  float32
}

type IF struct {
  prm 		*IFPrm
  n 		  int
  v 		  []float32
  ge 		  []float32
  gi 		  []float32
  cndr 	  []bool
  cndrc 	[]float64
  i 		  []float32
}

func (nrnm *IF) quant() int{
  return nrnm.n
}

func (nrnm *IF) history() []float64{
  return nrnm.cndrc
}

func (nrnm *IF) feed(in []float32) {
  nrnm.i = in
}

func (nrnm *IF) pulse() *[]bool {
  return &nrnm.cndr
}

func (nrnm *IF) G() *[]float32 {
  return &nrnm.ge
}

func (nrnm *IF) V() *[]float32 {
  return &nrnm.v
}

func (nrnm *IF) I() *[]float32 {
  return &nrnm.i
}

func (nrnm *IF) geti() (out []float32) {
  out = nrnm.i
  return
}

func (nrnm *IF) getv() (out []float32) {
  out = nrnm.v
  return
}

func (nrnm *IF) eeg(dt float32) {

  for i:=0; i<nrnm.n; i++ {
    nrnm.v[i]  += dt * (nrnm.ge[i] + nrnm.gi[i] - (nrnm.v[i] - nrnm.prm.el) + nrnm.i[i]) / nrnm.prm.tm
    nrnm.ge[i] += dt * -nrnm.ge[i] / nrnm.prm.te
    nrnm.gi[i] += dt * -nrnm.gi[i] / nrnm.prm.ti
  }

  for i:=0; i<nrnm.n; i++ {
    switch {
    case nrnm.v[i] >= nrnm.prm.vt:
      nrnm.cndr[i] = true
      nrnm.v[i] = nrnm.prm.vr
      nrnm.cndrc[i] += 1
    default:
      nrnm.cndr[i] = false
    }
  }
}

func if_init(n int, prm *IFPrm) *IF {
  return &IF{
    prm:  	prm,
    n:    	n,
    v:    	make([]float32, n),
    ge:   	make([]float32, n),
    gi:   	make([]float32, n),
    cndr: 	make([]bool, n),
    cndrc:	make([]float64, n),
    i:    	make([]float32, n),
  }
}

// NEURONS
// +++++++
// IF2

type IFEIPrm struct {
  tm      float32
  te      float32
  ti      float32
  vt      float32
  vr      float32
  el      float32
  ei      float32
  ee      float32
}

type IFEI struct {
  prm     *IFEIPrm
  n       int
  v       []float32
  ge      []float32
  gi      []float32
  cndr    []bool
  cndrc   []float64
  i       []float32
}

func (nrnm *IFEI) quant() int{
  return nrnm.n
}

func (nrnm *IFEI) history() []float64{
  return nrnm.cndrc
}

func (nrnm *IFEI) feed(in []float32) {
  nrnm.i = in
}

func (nrnm *IFEI) pulse() *[]bool {
  return &nrnm.cndr
}

func (nrnm *IFEI) G() *[]float32 {
  return &nrnm.ge
}

func (nrnm *IFEI) V() *[]float32 {
  return &nrnm.v
}

func (nrnm *IFEI) I() *[]float32 {
  return &nrnm.i
}

func (nrnm *IFEI) geti() (out []float32) {
  out = nrnm.i
  return
}

func (nrnm *IFEI) getv() (out []float32) {
  out = nrnm.v
  return
}

func (nrnm *IFEI) eeg(dt float32) {

  for i:=0; i<nrnm.n; i++ {
    nrnm.v[i]  += dt * (nrnm.ge[i] * (nrnm.prm.ee - nrnm.v[i]) + nrnm.gi[i] * (nrnm.prm.ei - nrnm.v[i]) - (nrnm.v[i] - nrnm.prm.el)) / nrnm.prm.tm
    nrnm.ge[i] += dt * -nrnm.ge[i] / nrnm.prm.te
    nrnm.gi[i] += dt * -nrnm.gi[i] / nrnm.prm.ti
  }

  for i:=0; i<nrnm.n; i++ {
    switch {
    case nrnm.v[i] > nrnm.prm.vt:
      nrnm.cndr[i] = true
      nrnm.v[i] = nrnm.prm.vr
      nrnm.cndrc[i] += 1
    default:
      nrnm.cndr[i] = false
    }
  }
}

func ifei_init(n int, prm *IFEIPrm) *IFEI {
  return &IFEI{
    prm:    prm,
    n:      n,
    v:      make([]float32, n),
    ge:     make([]float32, n),
    gi:     make([]float32, n),
    cndr:   make([]bool, n),
    cndrc:  make([]float64, n),
    i:      make([]float32, n),
  }
}


// NEURONS
// +++++++
// RATE

type RatePrm struct {

}

type Rate struct {
  prm       *RatePrm
  n         int
  x         []float32
  r         []float32
  g         []float32
  i         []float32
}

func (nrnm *Rate) quant() int{
  return nrnm.n
}

func (nrnm *Rate) history() []float64{
  return []float64{}
}

func (nrnm *Rate) feed(in []float32) {
  nrnm.i = in
}

func (nrnm *Rate) pulse() *[]bool {
  return &[]bool{}
}

func (nrnm *Rate) G() *[]float32 {
  return &nrnm.g
}

func (nrnm *Rate) V() *[]float32 {
  return &nrnm.r
}

func (nrnm *Rate) I() *[]float32 {
  return &nrnm.i
}

func (nrnm *Rate) geti() (out []float32) {
  out = nrnm.i
  return
}

func (nrnm *Rate) getv() (out []float32) {
  out = nrnm.r
  return
}

func (nrnm *Rate) eeg(dt float32) {
  for i:=0; i<nrnm.n; i++ {
    fmt.Printf("  g ( %v ) ", nrnm.g[i] + nrnm.i[i])
    nrnm.x[i] += dt * (-nrnm.x[i] + nrnm.g[i] + nrnm.i[i])
    nrnm.r[i] = math32.Tan(nrnm.x[i])
  }
  fmt.Printf("\n")
}

func rate_init(n int, prm *RatePrm) *Rate {
  xx := make([]float32, n)
  rr := make([]float32, n)
  for i:=0; i<n; i++ {
    xx[i] = rand.Float32() * 2 - 1
    rr[i] = math32.Tan(xx[i])
  }
  return &Rate{
    prm:    prm,
    n:      n,
    x:      xx,
    r:      rr,
    g:      make([]float32, n),
    i:      make([]float32, n),
  }
}



// ++++++++++++++++++++++++++++
// SYNAPSES


type Synapse interface {
  dojo(float32, float32)
  sim()
}


// SYNAPSES
// +++++++
// SPIKING SYNAPSES


type SSPrm struct {
  taui    float32
  tauo    float32
  wmax 	 	float32
  dai 	 	float32
  dao     float32
}

type SS struct {
  prm 		*SSPrm
  layerIn *Layer
  layerOut *Layer
  rptr 		[]int
  cptr 		[]int
  O 		  []int
  I 		  []int
  idx 	  []int
  w 		  []float32
  ti 		  []float32
  to 		  []float32
  ai 		  []float32
  ao 		  []float32
  cndro		*[]bool
  cndri 	*[]bool
  g 		  *[]float32
  i       *[]float32
}

func ss_init(i Layer, o Layer, d float32, p float32) *SS {

  rptr, cptr, O, I, idx, w := w_init(o.quant(), i.quant(), d, p)

  var taui float32 = 20
  var tauo float32 = 20
  var wmax float32 = 0.01
  dai := 0.01 * wmax

  return &SS{
    prm: &SSPrm{
      taui: taui,
      tauo: tauo,
      wmax: wmax,
      dai: dai,
      dao: -dai * taui / tauo * 1.05,
    },
    layerIn: &i,
    layerOut: &o,
    rptr: rptr,
    cptr: cptr,
    O: O,
    I: I,
    idx: idx,
    w: w,
    ti: make([]float32, len(w)),
    to: make([]float32, len(w)),
    ai: make([]float32, len(w)),
    ao: make([]float32, len(w)),
    cndro: o.pulse(),
    cndri: i.pulse(),
    g: o.G(),
    i: o.I(),
  }
}


func(ss *SS) fwd() {

  for i:=0; i<len(ss.cptr)-1; i++ {
    switch {
    case (*ss.cndri)[i] == true:
      k := ss.cptr[i]
      l := ss.cptr[i+1] -1
      for j := k; j < l+1; j++ {
        (*ss.g)[ss.O[j]] += ss.w[j]
      }
    }
  }
}


func(ss *SS) plasticity(dt, t float32) {
  fmt.Printf("  weights\n    - from %v\n", ss.w)
  for i:=0; i<len(ss.cptr)-1; i++ {
    switch {
    case (*ss.cndri)[i] == true:
      k := ss.cptr[i]
      l := ss.cptr[i+1] -1
      for j := k; j < l+1; j++ {
        ss.ai[j] *= e32( -(t - ss.ti[j]) / ss.prm.taui )
        ss.ao[j] *= e32( -(t - ss.to[j]) / ss.prm.tauo )
        ss.ai[j] += ss.prm.dai
        ss.ti[j]  = t
        ss.w[j]   = clamp(ss.w[j] + ss.ao[j], 0, ss.prm.wmax)
      }
    }
  }
  for i:=0; i<len(ss.rptr)-1; i++ {
    switch {
    case (*ss.cndro)[i] == true:
      k := ss.rptr[i]
      l := ss.rptr[i+1] -1
      for jt := k; jt < l+1; jt++ {
        j := ss.idx[jt]
        ss.ai[j] *= e32( -(t - ss.ti[j]) / ss.prm.taui )
        ss.ao[j] *= e32( -(t - ss.to[j]) / ss.prm.tauo )
        ss.ao[j] += ss.prm.dao
        ss.to[j]  = t
        ss.w[j]   = clamp(ss.w[j] + ss.ai[j], 0, ss.prm.wmax)
      }
    }
  }
  fmt.Printf("    - to  %v\n", ss.w)
}

func (ss *SS) dojo(dt, t float32) {
  ss.fwd()
  ss.plasticity(dt, float32(t))
}

func (ss *SS) sim() {
  ss.fwd()
}

// SYNAPSES
// +++++++
// SPARSE FORCE LEARNING


type SFLPrm struct {
}

type SFL struct {
  prm       *SFLPrm
  layerIn   *Layer
  layerOut  *Layer
  cptr      []int
  idx       []int
  synw      []float32
  ro        *[]float32
  ri        *[]float32
  g         *[]float32
  p         []float32
  q         []float32
  u         []float32
  outw      []float32
  f         float32
  z         float32
}

func sfl_init(in Layer, out Layer, p, g, lrnr float32) *SFL {
  nout := out.quant()
  nin  := in.quant()
  _, cptr, O, I, _, synw := wsfl_init(nout, nin, p, g)

  u := make([]float32, nout)
  outw := make([]float32, nout)
  for i:=0; i<nout; i++ {
    u[i] = 2 * rand.Float32() - 1
    outw[i] = 1/math32.Sqrt(float32(nout)) * (2 * rand.Float32() - 1)
  }

  pp := make([]float32, len(I))
  for i, _ := range I {
    switch {
    case I[i] == O[i]:
      pp[i] = lrnr
    }
  }

  return &SFL{
    prm: &SFLPrm{},
    layerIn: &in,
    layerOut: &out,
    cptr: cptr,
    idx: O,
    synw: synw,
    ro: out.V(),
    ri: in.V(),
    g: out.G(),
    p: pp,
    q: make([]float32, nout),
    u: u,
    outw: outw,
  }
}


func(sfl *SFL) fwd() {

  sfl.z = dot(sfl.outw, *sfl.ro)
  (*sfl.g) = mulbdcst(sfl.z, sfl.u)

  sfl.q = make([]float32, (*sfl.layerOut).quant())
  for i:=0; i<len(sfl.cptr)-1; i++ {
    s := (*sfl.ri)[i]
    k := sfl.cptr[i]
    l := sfl.cptr[i+1]
    for jt := k; jt < l; jt++ {

      j := sfl.idx[jt]
      sfl.q[j]    += sfl.p[jt] * s
      (*sfl.g)[j] += sfl.synw[jt] * s
      fmt.Printf("  G ( %v ) \n", (*sfl.g))

    }
  }
}


func(sfl *SFL) plasticity(dt, t float32) {

  c := 1 / ( 1+ dot(sfl.q, *sfl.ro))

  axpy(c * (sfl.f - sfl.z), sfl.q, sfl.outw)

  for i:=0; i<len(sfl.cptr)-1; i++ {
    k := sfl.cptr[i]
    l := sfl.cptr[i+1]
    for j := k; j < l; j++ {

      // sfl.p[j] += -c * sfl.q[sfl.idx[j]] * sfl.q[j]
      sfl.p[j] += -c * sfl.q[sfl.idx[j]] * sfl.q[i]

    }
  }
}

func (sfl *SFL) dojo(dt, t float32) {

  sfl.fwd()

  sfl.plasticity(dt, float32(t))

}

func (sfl *SFL) sim() {
  sfl.fwd()
}


// SYNAPSES
// +++++++
// ALL2ALL FORCE LEARNING


type A2AFLPrm struct {
}

type A2AFL struct {
  prm       *A2AFLPrm
  layerIn   *Layer
  layerOut  *Layer
  synw      [][]float32
  ro        *[]float32
  ri        *[]float32
  g         *[]float32
  p         [][]float32
  q         []float32
  u         []float32
  outw      []float32
  f         float32
  z         float32
}

func fl_init(in Layer, out Layer, p, g, lrnr float32) *A2AFL {
  nout := out.quant()
  nin  := in.quant()
  w, outw := wfl_init(nout, nin, p, g)

  u := make([]float32, nout)
  for i:=0; i<nout; i++ {
    u[i] = 2 * rand.Float32() - 1
  }

  pp := idmat(nout, lrnr)
  fmt.Printf("  GGG\n    - %v\n", out.G())

  return &A2AFL{
    prm: &A2AFLPrm{},
    layerIn: &in,
    layerOut: &out,
    synw: w,
    ro: out.V(),
    ri: in.V(),
    g: out.G(),
    p: pp,
    q: make([]float32, nout),
    u: u,
    outw: outw,
  }
}


func(fl *A2AFL) fwd() {

  fl.z = dot(fl.outw, *fl.ro)
  fl.q = mul(fl.p,  *fl.ri)
  *fl.g = mul(fl.synw,  *fl.ri)
  axpy(fl.z, fl.u, *fl.g)

}


func(fl *A2AFL) plasticity(dt, t float32) {

  c := 1 / ( 1+ dot(fl.q, *fl.ro))

  axpy(c * (fl.f - fl.z), fl.q, fl.outw)

  q := blas32.Vector{
    Inc:  1,
    Data: fl.q,
  }
  data := make([]float32, 0)
  for _, v := range fl.p {
    data = append(data, v...)
  }
  p := blas32.General{
    Rows:   len(fl.p),
    Cols:   len(fl.p[0]),
    Stride: len(fl.p),
    Data:   data,
  }
  blas32.Ger(-c, q, q, p)


  for i, v := range p.Data {
    x := i / p.Cols
    y := i % p.Cols
    fl.p[x][y] = v
  }
}

func (fl *A2AFL) dojo(dt, t float32) {
  fl.fwd()
  fl.plasticity(dt, float32(t))
}

func (fl *A2AFL) sim() {
  fl.fwd()
}


// ++++++++++++++++++++++++++++
// DOJO

type Model struct {
  layers    *[]Layer
  synapses  *[]Synapse
}


func (mdl *Model) train(dt, d float32) {

  timesteps, _ := mktmstps(d, dt)
  for i, t := range timesteps {
    fmt.Printf("Train step %v at %v\n", i, t)
    mdl.train2(dt, t)
  }
}

func (mdl *Model) train2(dt float32, t float32) {

  for _, layer := range (*mdl.layers) {
    layer.eeg(dt)
  }

  for _, synapse := range (*mdl.synapses) {
    synapse.dojo(dt, t)
  }
}

func (mdl *Model) sim(dt, d float32) {

  timesteps, _ := mktmstps(d, dt)
  for i, t := range timesteps {
    fmt.Printf("Sim step %v at %v\n", i, t)
    mdl.sim2(dt)
  }
}

func (mdl *Model) sim2(dt float32) {

  for _, layer := range (*mdl.layers) {
    layer.eeg(dt)
  }
  for _, synapse := range (*mdl.synapses) {
    synapse.sim()
  }
}


func main() {

  rand.Seed(time.Now().UTC().UnixNano())

  // izp := IZPrm{1,2,3,4,30}
  // iz0 := iz_init(4, &izp)
  // iz1 := iz_init(4, &izp)

  // ifp := IFPrm{20, 5, 10, 30, -60, -60}
  // if0 := if_init(4, &ifp)
  // if1 := if_init(4, &ifp)

  // ifeip := IFEIPrm{20, 5, 10, -50, -60, -60, 0, 0}
  // if0 := ifei_init(4, &ifeip)
  // if1 := ifei_init(4, &ifeip)

  ratep := RatePrm{}
  r0 := rate_init(4, &ratep)
  r1 := rate_init(4, &ratep)

  // fmt.Println(if0)

  // tic := time.NewTicker(5*time.Second)


  layers := make([]Layer,0)
  // layers = append(layers, iz0)
  // layers = append(layers, iz1)
  // layers = append(layers, if0)
  // layers = append(layers, if1)
  layers = append(layers, r0)
  layers = append(layers, r1)


  synapses := make([]Synapse, 0)
  // ss01 := ss_init(layers[0], layers[1], 2.0, 1.0)
  // synapses = append(synapses, ss01)
  // sfl01 := sfl_init(layers[0], layers[1], 1.0, 1.5, 0.01)
  fl01 := fl_init(layers[0], layers[1], 1.0, 1.5, 0.01)
  synapses = append(synapses, fl01)

  var model Model
  model.layers = &layers
  model.synapses = &synapses


  d := 1.0 * float32(ms)
  dt :=  float32(ms) / 10
  timesteps, n := mktmstps(d, dt)

  sincomp := make([]float32, n)
  results := make([][]float32, n)
  for i, t := range timesteps {
    fmt.Printf("Training step %v at %v\n", i, t)
    sincomp[i] = math32.Sin(t)
    fmt.Printf(" * TARGET = %v\n", sincomp[i])
    sfl01.z = sincomp[i]
    model.train2(float32(dt), t)
    // model.sim2(float32(dt))
    results[i] = (*model.layers)[1].getv()
    fmt.Printf(" * OUTPUT = %v\n", results[i])
  }


  // plotting(lyr, tic)
  // time.Sleep(2 * time.Second)
  // iz.cndrc[1] +=1
  // iz.cndrc[1] +=2
  // plotting(lyr, tic)
  // tic.Stop()

}





// ++++++++++++++++++++++++++++
// UTILS

// UTILS
// +++++++
// TIMESTEPS

func mktmstps(d, dt float32) (out []float32, n int) {
  n = int(d / dt) + 1
  fmt.Printf("d %v", d)
  fmt.Printf("dt %v", dt)
  for i:=0; i<n; i++{
    out = append(out, float32(i)*dt)
  }
  return
}

// UTILS
// +++++++
// ALGEBRA

func e32(in float32) (out float32) {
  switch {
  case in < -10:
    out = -32
  default:
    out = in
  }
  out = 1 + out/32
  out *= out
  out *= out
  out *= out
  out *= out
  out *= out
  return
}

func clamp(x, min, max float32) (out float32) {
  switch {
  case (x >= min) && (x <= max):
    out = x
  case (x < min):
    out = min
  case (x > max):
    out = max
  }
  return
}

func dot(x []float32, y []float32) (dot float32) { 
  for i, v := range x { 
    dot += v * y[i]
  } 
  return
}

func mul(x [][]float32, y []float32) (out []float32) {

  out = make([]float32, len(y))
  for i, _ := range x {
    for j, _ := range x[i] {
      out[i] += x[i][j] * y [j]
    }
  } 
  return
}

func mulbdcst(x float32, y []float32) (out []float32) {

  out = make([]float32, len(y))
  for i, yi := range y {
    out[i] = x * yi
  } 
  return
}

func axpy(a float32, x, y []float32) { 

  for i, _ := range y { 
    y[i] += a * x[i]
  } 
  return
}

func makesin(a,f, t float32) float32 {

  var pi float32 = 3.1416
  return a * math32.Sin(pi * f * t) + a/2 * math32.Sin(2*pi * f * t) + a/6 * math32.Sin(3*pi * f * t) + a/3 * math32.Sin(4*pi * f * t)
}

func idmat(n int, lrn float32) [][]float32 {

  id := make([][]float32, n)
  for i:=0; i<n; i++ {
    tmp := make([]float32, n)
    for j:=0; j<n; j++ {
      switch {
      case j == i:
        // tmp[j] = 1/lrn
        tmp[j] = lrn
      default:
        tmp[j] = 0
      }
    }
    id[i] = tmp
  }
  return id
}

// UTILS
// +++++++
// WEIGHTS JANITORING

func sprand(m,n int, d, p float32) (w [][]float32) {

  for i:=0; i<m; i++ {   // COLUMNS
    tmp := make([]float32, 0)
    for j:=0; j<n; j++ { // ROWS
      coin := rand.Float32()
      switch {
      case coin < float32(p):
        toss := d * rand.Float32()
        tmp = append(tmp, toss)
      default:
        tmp = append(tmp, 0)
      }
    }
    w = append(w, tmp)
  }

  return
}

func sprandsfl(m,n int, p, g float32) (w [][]float32) {

  scale := g / math32.Sqrt(p * float32(m))
  for i:=0; i<m; i++ {   // COLUMNS
    tmp := make([]float32, 0)
    for j:=0; j<n; j++ { // ROWS
      coin := rand.Float32()
      switch {
      case coin < float32(p):
        toss := scale * rand.Float32()
        tmp = append(tmp, toss)
      default:
        tmp = append(tmp, 0)
      }
    }
    w = append(w, tmp)
  }
  return
}

func sprandfl(m,n int, g float32) (w [][]float32) {

  scale := g / math32.Sqrt(float32(m))
  for i:=0; i<m; i++ {   // COLUMNS
    tmp := make([]float32, 0)
    for j:=0; j<n; j++ { // ROWS
      toss := scale * float32(rand.NormFloat64())
      tmp = append(tmp, toss)
    }
    w = append(w, tmp)
  }
  return
}

func transpose(in [][]float32) (out [][]float32) {

  m := len(in)
  n := len(in[0])
  out = make([][]float32, n)
  for i := 0; i < n; i++ {
    out[i] = make([]float32, m)
  }
  var row []float32
  for i := 0; i < n; i++ {
    row = out[i]
    for j := 0; j < m; j++ {
      row[j] = in[j][i]
    }
  }
  return
}

func w_init(m,n int, d, p float32) ([]int, []int, []int, []int, []int, []float32) {	

  w := sprand(m, n, d, p)
  wt := transpose(w)

  spw := toCSC(w)
  spwt := toCSC(wt)
  cptr := spw.cptr
  rptr := spwt.cptr
  index := make([]int, len(spw.i))
  cdown := make([]int, len(cptr)-1)

  for i := 0; i < len(rptr)-1; i++ {
    k := cptr[i]
    l := cptr[i+1]
    for x := k; x < l; x++ {
      j := spwt.i[x]
      index[x] = cptr[j] + cdown[j]
      cdown[j] += 1
    }
  }
  return rptr, cptr, spw.i, spw.j, index, spw.data
}

func wsfl_init(m,n int, p, g float32) ([]int, []int, []int, []int, []int, []float32) { 

  w := sprandsfl(m, n, p, g)
  // w := [][]float32{{0.972005, 0.35119, 0.0700439, 0.961996},
  //                  {0.50784, 0.132046, 0.945162, 0.694922},
  //                  {0.529964, 0.0730582, 0.690167, 0.760609},
  //                  {0.442268, 0.0579783, 0.535791, 0.359336},
  //                }
  wt := transpose(w)

  spw := toCSC(w)
  spwt := toCSC(wt)
  cptr := spw.cptr
  rptr := spwt.cptr
  index := make([]int, len(spw.i))
  cdown := make([]int, len(cptr)-1)

  for i := 0; i < len(rptr)-1; i++ {
    k := cptr[i]
    l := cptr[i+1]
    for x := k; x < l; x++ {
      j := spwt.i[x]
      index[x] = cptr[j] + cdown[j]
      cdown[j] += 1
    }
  }

  // fmt.Printf("  ====== INDEX\n    - %v\n", index)
  // fmt.Printf("         len  :  %v\n", len(index))
  // fmt.Printf("  ====== WEIGHTS\n    - %v\n", spw.data)
  // fmt.Printf("         len  :  %v\n", len(spw.data))  
  return rptr, cptr, spw.i, spw.j, index, spw.data
}

func wfl_init(m,n int, p, g float32) ([][]float32, []float32) { 

  w := sprandfl(m, n, g)
  wout := make([]float32, m)
  scale := 1 / math32.Sqrt(float32(m))
  for i := 0; i < m; i++ {
    wout[i] = scale * float32(rand.NormFloat64())
  }

  return w, wout
}

// UTILS
// +++++++
// SPARSE CSC

type Sparse struct {
  m       int
  n       int
  cptr    []int
  ptr     []int
  i       []int
  j       []int
  data    []float32
}

func toCSC(in [][]float32) Sparse {

  m := len(in)
  n := len(in[0])
  var c int
  cptr := make([]int, 0)
  ptr := make([]int, 0)
  data := make([]float32, 0)
  ii := make([]int, 0)
  jj := make([]int, 0)
  for j:=0; j<n; j++ {   // COLUMNS
    tmp := make([]int, 0)
    for i:=0; i<m; i++ { // ROWS
      switch {
      case in[i][j] == 0:
        continue
      default:
        tmp = append(tmp, i)
        data = append(data, in[i][j])
        ii = append(ii, i)
        jj = append(jj, j)
      }
    }
    ptr = append(ptr, tmp...)
    cptr = append(cptr, c)
    c += len(tmp)
  }
  cptr = append(cptr, c)
  return Sparse{m:m, n:n, cptr:cptr, ptr:ptr, i:ii, j:jj, data:data}
}



// UTILS
// +++++++
// PLOTTING


func plotting(lyrs []Layer, tic chan uint8) {

  if err := ui.Init(); err != nil {
    log.Fatalf("failed to initialize termui: %v", err)
  }
  defer ui.Close()
  p := widgets.NewParagraph()  
  p.Title = "Press q to quit"
  p.Text = "Neuromorphic network"
  p.SetRect(0, 0, 50, 5)
  p.TextStyle.Fg = ui.ColorWhite
  p.BorderStyle.Fg = ui.ColorCyan

  widg := make([]*widgets.BarChart, len(lyrs))
  height := 10
  width := 100
  x0 := 0

  for i, lyr := range lyrs {
    widg[i] = widgets.NewBarChart()
    widg[i].Title = fmt.Sprintf("Layer %v", i)
    widg[i].Data = lyr.history()
    widg[i].SetRect(x0, x0+height*i, x0+width, x0+height*(i+1))
  }

  draw := func() {
    for i, w := range widg {
      w.Data = lyrs[i].history()
      ui.Render(w)
    }
  }

  draw()
  uiEvents := ui.PollEvents()
  for {
    select {
    case e := <-uiEvents:
      switch e.ID {
      case "q", "<C-c>":
        return
      }
    case <-tic:
      draw()
    }
  }
  return
}
