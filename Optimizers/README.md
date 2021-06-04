# Optimizers

## Gradient Descent
* Intuition: Roll a ball downwards to reach the minima.
* Take repeated steps in the opposite direction of the graient of the loss at the current point ( direction of the steepest descent ) to reach the minima.
* Note when the loss curve is steep the gradient is large and when the curve is gentle the gradient is small.
* Implies the updates are large in the areas where curve is steep and updates are small where the curve is gentle.
* Disadvantage: Once we hit a surface which has gentle slope ( plateau region ) the progress slows down.

## Momentum based Gradient Descent
* Intuition: Ball gains momentum while rolling down a slope.
* Extension of GD: In addition to the current update, also look at the previous updates.
* Even in plateau regions, MBGD is able to take large steps because the momentum carries it along.
* Disadvantage: Oscillates in and out of the minima valley as the momentum carries it out of the valley.
