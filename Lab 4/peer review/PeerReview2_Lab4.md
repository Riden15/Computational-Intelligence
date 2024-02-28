# Peer Review
Peer review for Lab 10 of [Arman Behkish s299525](https://github.com/ArmanBehkish/computational-intelligence_2324/tree/master/2023-24/LAB%2010)

## Intro
Hi Arman ✌️.

Firstly, good work 👍🏻! You have written a nice and working RL algorithm.
The code is also well written, you added a small description to each function plus some comments in the code. 
This is always well appreciated. You also added a `README` file which explains the strategy used very well so excellent!
To make it even more complete, you could have added some results that you obtained while testing your algorithm.

## Code
To improve your algorithm I can offer you a couple of ideas:
1. I don't know how many different learning rates you've tried, but it's definitely a good starting point 
to start testing your algorithm with different values and see if anything changes.
2. What can give a big boost to the algorithm is to add more state exploration. The simplest thing that could be done 
is to ensure that during training there is a probability (maybe 20% or 30%) that the algorithm chooses a random move 
instead of consulting the `value_dictionary`. To improve even more, you could make the exploration rate start from 1 
and lower more and more as the epochs increase.
3. Lastly, you could try adding the discount factor to the Bellman equation. The discount factor essentially determines 
how much the reinforcement learning agents care about rewards in the distant future relative to those in the immediate future.
With this, the formula will be:
```python
value_dictionary[hashable_state] = value_dictionary[hashable_state] + epsilon * (discount_factor * final_reward - value_dictionary[hashable_state])
```

## Ending
I hope you'll find this review useful and good luck for the project 😉!