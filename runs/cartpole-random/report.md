# Experiment report: cartpole-random

## Configuration

| key | value |
|---|---|
| env | `gym:CartPole-v1` |
| agent | `random` |
| seed | 7 |
| episodes | 5 |
| max_steps | 200 |
| eval_episodes | 3 |

## Result

- final state: **DONE**
- wall time: 0.2s
- eval mean reward: **18.00** (std 6.16)
- eval min/max: 10.00 / 25.00
- success rate: 100%
- training reward 16.0 -> 13.0 (flat/declined)

## Learning curve

![learning curve](learning_curve.png)
