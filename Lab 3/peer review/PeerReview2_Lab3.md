# Peer Review
Peer review for Lab 3 of [Yalda sadat Mobargha s314700](https://github.com/YaldaMobargha/Computational-intelligence/tree/main/Labs/9)

## Intro
Hi Yalda ✌️.
Firstly, good job👍🏻! Very good also to have added the `README.md` file, it is very useful for immediately 
understanding the work done without going into detail.
The code is pretty simple to read, however, you could use some comments, they are always nice to have.

## Code
The code you committed doesn't run due to a small error in the `mutate` function. 
From what I understand, you invert a random individual of the given input population.
If you change your genome from a tuple to a simple array and change the function mutate like this, it should work:
```python
def mutate(genome):
    index = randint(0, NUM_LOCI-1)
    genome[index] = 1-genome[index]
    return genome
```

## Possible improvements
I am of the opinion that incorporating various strategies with diversity and expanding the generational scope could significantly improve outcomes. 
Moreover, the application of diversity promotion is likely to resolve problem instance 1 within the existing generation count.

## Ending
I hope you'll find this review useful and good luck for the next labs 😉!