# YOUR CODE HERE
a = list(map(int, input().split()))
b = list(map(int, input().split()))

takahashi_score = sum(a)
aoki_score = sum(b)

print(takahashi_score - aoki_score + 1)