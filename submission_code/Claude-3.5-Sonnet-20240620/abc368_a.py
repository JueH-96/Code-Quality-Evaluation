# YOUR CODE HERE
N, K = map(int, input().split())
cards = list(map(int, input().split()))

# Rotate the list by K positions
rotated_cards = cards[-K:] + cards[:-K]

# Print the rotated cards
print(*rotated_cards)