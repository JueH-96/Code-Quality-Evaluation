# Read the input from stdin
import sys

# Read the number of cards and the number of cards to move
N, K = map(int, sys.stdin.readline().split())

# Read the cards
cards = list(map(int, sys.stdin.readline().split()))

# Move the cards
moved_cards = cards[-K:] + cards[:-K]

# Print the result
print(' '.join(map(str, moved_cards)))