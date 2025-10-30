# YOUR CODE HERE

import sys

N, K = map(int, sys.stdin.readline().split())
cards = list(map(int, sys.stdin.readline().split()))

cards = cards[-K:] + cards[:-K]

print(' '.join(map(str, cards)))