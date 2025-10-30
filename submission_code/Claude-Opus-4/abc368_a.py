# YOUR CODE HERE
N, K = map(int, input().split())
cards = list(map(int, input().split()))

# Take K cards from bottom (last K elements) and place on top
result = cards[-K:] + cards[:-K]

# Print the result
print(' '.join(map(str, result)))