# YOUR CODE HERE
n, k = map(int, input().split())
cards = list(map(int, input().split()))
bottom_cards = cards[-k:]
top_cards = cards[:-k]
result = bottom_cards + top_cards
print(*result)