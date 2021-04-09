# Kalman Filter

Excellent [videos](https://www.youtube.com/playlist?list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT)

## Elements & High Level Flow

Given some values (aka **state**) we want to track, the key items are:

Given starting point of:

- State matrix (S): previous estimates of values we are tracking, or initial estimate.
  This represents what the kalman filter is trying to track.
- Measurement Errors (MErr):observations have errors, not perfect. For physical process
  the error may be a constant linked to the measurement device.
- Error in prediction (EErr): estimated error in the estimate / prediciton

1. Measurements (M): values based on observations.
2. Kalman Gain (KG): calculated based on MErr and Previous Error in state estimate
   / prediction (EErr).
3. Estimated / Predicted NEW state matrix (EST): calculated from
   - previous S, based on a user defined prediction model. E.g. a physics equation
   - current M
4. Update Error in prediction matrix (EErr): calculated from EST and KG.
   This is used to estimate Kalman Gain for the next iteration.

## Kalman Gain & Prediction

Conceptually, $0 <= KG <= 1$:

$$ KG = \frac{EErr}{EErr + MErr} $$

Generally, the prediction for the new state matrix is:

$$ EST_t = EST_{t-1} + KG [M - EST_{t-1}] = KG \cdot M + (1 - KG) \cdot EST_{t-1} $$

When prediction error is large vs MErr, KG is closer to 1, the prediction favours
new measurement.

When prediction error is small vs MErr, KG is closer to 0, the prediction favours
the system's defined prediction model.

## Multi-Dimensional Representation

- $k$ - iternation $k$
- $N$ state variables to track
- $X_k$ - state matrix at step $k$, $N \times 1$
- $P_k$ - process covariance matrix / Error in prediction (EErr), $N \times N$.
  Initial value given by user.
- $Q$ - process noice covariance matrix, $N \times N$
- $K$ - Kalman Gain, $N \times N$
- $Y$ - Measurement of the state, same dimension as $X_k$, $N \times 1$
- $Z$ - Measurement noice (MErr), same dimension as $Y$, $N \times 1$
- $R$ - Measurement noice covariance matrix, $N \times N$
- $I$ - Identity matrix, $N \times N$
- $u$ - Control Variable Matrix - prediction model, $V \times 1$, it doesn't
  need to be the same shape as $X$
- $w$ - Predicted state noise matrix - prediction model, $N \times 1$
- $A$, $B$ - prediction model to convert previous state to next state, $N \times N$ and $N \times V$
- $H$ - Kalman gain model, $N \times N$

1. Measurement input:

$$ Y_k = C \cdot X_{k_m} + Z_k $$

2. Compute predicted state based on model

$$ X_{k_p} = A X_{k-1} + B u_{k} + w_k $$

3. Compute predicted Error in prediction based on model

$$ P_{k_p} = A P_{k-1} A^T + Q_k $$

4. Compute Kalman Gain

$$ K = \frac{P_{k_p} H}{H P_{k_p} H^T + R} $$

5. Compute predicted state, incorporating measurement

$$ X_k = X_{k_p} + K[Y - H X_{k_p}] $$

6. Compute new Error in the prediciton process (EErr)

$$ P_k = (I - KH) P_{k_p} $$
