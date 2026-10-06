# Experiment report: etch-qlearning-v1_seed1

## Configuration

| key | value |
|---|---|
| env | `etch_chamber` |
| agent | `qlearning` |
| seed | 1 |
| episodes | 300 |
| max_steps | 100 |
| eval_episodes | 20 |

## Result

- final state: **IDLE**
- wall time: 0.0s
- eval mean reward: **-3.79** (std 0.00)
- eval min/max: -3.79 / -3.79
- success rate: 100%
- training reward -57.3 -> -4.5 (improved)

## Learning curve

![learning curve](learning_curve.png)
