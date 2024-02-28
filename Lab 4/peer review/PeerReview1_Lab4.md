# Peer Review
Peer review for Lab 10 of [Davide Sferrazza s326619](https://github.com/FarInHeight/Computational-Intelligence/tree/main/lab10)

## Intro
Hi Davide ✌️.

Firstly, good job 👍🏻! Each function is well documented and the code is easy to read. Your players also seem unbeatable, nice!
Also, nice job on adding to practically every single line in the notebook a little comment to describe what's happening, 
it gave me a huge help in understanding the code and therefore saved me some time.
I also discovered that it's possible to print the board using some emoji for ❌ and ⭕️ and I must say that it turns out 
really well, I will definitely use it in the future 😏.

## Code

While reading the code, I just noticed a small typo that's also quite irrelevant but anyway, in the ``_game_reward`` 
method you write that the player is a `TicTacToe` class instead of a `QLearningRLPlayer`.

Since you use the state in the form of a string as the dictionary key, it took me a while to understand the whole process 
between wanting to change the -1 to 2 of the board state in `_map_state_to_index` to the various reshapes in `_make_move`
and immediately after to get the `action`. I personally find working with strings quite uncomfortable and often not very
intuitive, which is why I opted to transform the board matrix into a tuple of tuples and use it as a key. For example, a matrix 
[[1,0,0], [0,1,0], [0,0,1]] will become ((1,0,0), (0,1,0), (0,0,1)). In this way, in my opinion, it is slightly easier 
to work with and also more intuitive.

Changing the exploration rate at each epoch is also excellent. I think this step improves the player and not just a little. 
I hadn't really thought of this, so thank you very much for this addition, I will definitely use it for the project.

## Ending
I hope you'll find this review useful and good luck for the project 😉!