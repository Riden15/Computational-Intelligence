# Peer Review
Peer review for Lab 2 of [Luca Sturaro s320062](https://github.com/HerryTheBest/Polito---Computational-Intelligence/tree/main/Lab_2)

## Intro
Hi Luca ✌️.

First of all, it would have been easier to evaluate the code if the README.md file had included some information about
your ES. For example, describe the association between the percentage parameter and the choice of row in ``evolutionary_strategy_1``. 
It seems to me that a low percentage corresponds to prefer the first rows and a high percentage corresponds to prefer the last rows. 

I'd like to divide my review into two parts corresponding on your two strategies:
- ``evolutionary_strategy_1`` and ``evolve_1``
- ``evolutionary_strategy_2`` and ``evolve_2``


## Relative position strategy
Firstly, In ``evolutionary_strategy_1`` I think that's simply a typo. You never use the parameter that indicates how many elements from that row to take.
```python
num_objects = ceil(choice_parameters[0] / 100 * state.rows[row])
```

In this solution, if I understand correctly, there is only an individual in the population that is ``ev_parameters``.
This individual plays with the optimal strategy and, depending on the number of wins, it can become the ``best_ev_parameters``.
After that, you apply a mutation on that individual and the cycle starts again.

I personally implement a similar solution, where an individual is evaluated by testing him against the optimal algorithm.
However, the problem with your solution is that that's not an ES. Without a population and a parent selection, you are doing
a local search, and so you might get stuck in a local optimum.

## Ideal Nim-sum strategy
For this implementation, you used the nim-sum as fitness, although I don't think it's a correct evolution strategy (assuming I understood your intentions).
The process involves generating an offspring, in the ``evolutionary_strategy_2`` function, and verify the fitness for every individual creates, one by one.
This offspring, composed of moves, determines the best move to execute based on the nim-sum value as the fitness criterion. A smaller nim-sum indicates a superior move.
This approach ensures that the only real-valued parameter undergoing training is the variable ``ev_parameter``.

I expected you to develop an agent with the ability to play independently, eliminating the necessity to generate an offspring solely for executing a single move.

## Ending
I hope you'll find this review useful and good luck for the next labs 😉!