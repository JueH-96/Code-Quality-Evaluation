n, k = map(int, input().split())
A = list(map(int, input().split()))

# Take the last k cards and place them on top
top_cards = A[-k:]
rest_cards = A[:-k]

# Concatenate the top cards with the rest
result = top_cards + rest_cards

print(*result)