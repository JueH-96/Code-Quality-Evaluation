# YOUR CODE HERE
n, k = map(int, input().split())
a = list(map(int, input().split()))

moved_cards = a[n - k:]
remaining_cards = a[:n - k]

result = moved_cards + remaining_cards
print(*result)