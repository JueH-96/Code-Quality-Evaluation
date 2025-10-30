# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Take K cards from the bottom (last K elements)
bottom_cards = A[-K:]
# Remaining cards (first N-K elements)
remaining_cards = A[:-K]

# Place bottom cards on top, followed by remaining cards
result = bottom_cards + remaining_cards

print(*result)