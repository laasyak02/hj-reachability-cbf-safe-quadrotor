# Reachability-Based Safe Control for Quadrotors

A safety filter that keeps a quadrotor inside a provably safe set during flight, built with **Hamilton–Jacobi (HJ) reachability** and **Control Barrier Functions (CBFs)**, and enforced online through an **MPC safety filter**. The nominal MPC controller tracks the trajectory freely and the safety filter overrides its command only when needed to stay safe.

Demonstrated in simulation and on a real Crazyflie, with a comparison of two safety-filter formulations (detailed below).

> **Note:** This repo presents the method and results. The code is part of ongoing work being prepared for publication, and is therefore not included here.
---

## Why HJ reachability?

A safety filter can only guarantee safety if it knows which states are actually safe — i.e. the states from which a collision can still be avoided given the vehicle's dynamics and limited control authority. HJ reachability computes exactly that: it produces a value function whose sign certifies safety at every state, under the full nonlinear dynamics and actuation limits.

The catch is the **curse of dimensionality** — solving it directly over the full 10D quadrotor state is intractable. Therefore, the system is **decomposed into lower-dimensional subsystems**, solved for reachability on each, and recombined together.

---

## Method

### System model

A **10D near-hover quadrotor model**:

```
state (10D) = (px, vx, θx, ωx,  py, vy, θy, ωy,  pz, vz)
control (3) = (Sx, Sy, Tz)
```

Position/velocity per axis, plus pitch/roll angle and rate. Controls are the commanded attitude angles `Sx, Sy` (which steer lateral motion) and collective thrust `Tz`. Under the near-hover assumption the lateral (x, y) and vertical (z) channels decouple, which makes the decomposition possible.

### Decomposed reachability

Instead of one intractable 10D solve, the system is split into three decoupled subsystems and HJ reachability is solved on each:

| Subsystem | State | Input | Bound |
|-----------|-------|-------|-------|
| **x** | pos, vel, pitch, pitch rate | `Sx` | \|Sx\| ≤ 10° |
| **y** | pos, vel, roll, roll rate | `Sy` | \|Sy\| ≤ 10° |
| **z** | altitude, vertical vel | `Tz` | 0 ≤ Tz ≤ 2g |

Each gives a value function `h_x, h_y, h_z`, positive wherever that axis alone keeps the vehicle clear of the obstacles. These are precomputed offline and looked up at run time.

### Combining the barriers

A box obstacle is cleared if the quadrotor is safe along **at least one** axis (go around in x, or y, or over in z). That OR is a **max**:

```
h = max(h_x, h_y, h_z)
```

The MPC filter needs `h` *and* its gradient. A literal `max` is non-differentiable where two axes tie — its gradient jumps there, causing the filter to **chatter** between axes. This repo implements two ways to handle that:

**1. Hard-max** (Glotfelter et al., CCTA 2018) — enforces a CBF constraint for every axis within an ε-band of the max, keeping the constraint set continuous across a switch. Exact and cheap, but control can still jump near ties.

**2. Log-sum-exp (LSE)** — a smooth soft-max:
```
h = (1/α) · log Σ exp(α · h_i) − log(3)/α
```
Smooth and differentiable everywhere, so no chatter. The `−log(3)/α` term matters for safety: a plain LSE *over-estimates* the max, which would make the safe set look bigger than it is (unsafe); the correction cancels that overshoot so the barrier never exceeds the true safe set. The gradient is a softmax-weighted blend of the per-axis Lie derivatives (Molnar & Ames). `α → ∞` recovers hard-max; smaller `α` is smoother but more conservative.

### Control loop

Three stages per step; the nominal tracker and the safety filter are **independent solves**:

1. **Nominal MPC tracker** → `u_nom = (Sx, Sy, Tz)`
2. **CBF safety filter** — solves `min ‖u − u_nom‖²` s.t. CBF rows ≥ 0 → `u_safe` *(hard-max or LSE applied here)*
3. **Low-level attitude/thrust controller** — tracks `u_safe`

---

## Results

The results of both methods are demonstrated in simulation (PyBullet) as well as on real hardware.

### Simulation (PyBullet)

Both methods are run on the **same obstacle layouts**, so any difference comes from the composition method alone.

| Hard-max | Log-sum-exp |
|----------|-------------|
| *[hard-max sim video]* | *[LSE sim video]* |

The hard-max method demonstrates a lot more chattering, due to constant jumps in the control input near the obstacle barrier, as compared to the smooth log-sum-exp composition method. This difference can be observed in the demonstration videos: the hard-max method shows jerky behavior while avoiding the obstacles, whereas the LSE method avoids them smoothly due to the almost-constant safety-filter activation.

### Hardware — Crazyflie 2.x

Deployed on a real Crazyflie with Vicon motion capture. The videos below show the behavior of the drone under the two different composition techniques for the same obstacle layout.

| Hard-max | Log-sum-exp |
|----------|-------------|
| https://github.com/user-attachments/assets/1abb0d93-2149-4bea-a372-832f6a237acf | https://github.com/user-attachments/assets/e30ee198-2d4d-4bc9-9a33-144106c59e73 |

The drone hits an obstacle under the hard-max composition, while successfully avoids the obstacles under the smooth LSE composition.

Below are a few more examples using the LSE method, with different obstacle configurations, where the drone successfully avoids the obstacles during its traversal.

| | | |
|--|--|--|
| https://github.com/user-attachments/assets/361f7581-4ced-4183-8a61-47a2a1ab974a | https://github.com/user-attachments/assets/a0aa9150-7b83-4273-a428-5ed74a0e3b72 | 

---

## Stack

`HJ reachability` · `Control Barrier Functions` · `MPC (acados)` · `CasADi` · `JAX` · `PyBullet (gym-pybullet-drones)` · `ROS2` · `Docker` · `Crazyflie 2.x` · `Vicon` · `Python`

---

## References

- Glotfelter, Cortés, Egerstedt. *Boolean Composability of Constraints and Control Synthesis for Multi-Robot Systems via Nonsmooth Control Barrier Functions.* CCTA, 2018.
- Glotfelter, Cortés, Egerstedt. *Nonsmooth Barrier Functions with Applications to Multi-Robot Systems.* IEEE L-CSS, 2017.
- Ames et al. *Control Barrier Function Based Quadratic Programs with Application to Adaptive Cruise Control.* CDC, 2014.

---
