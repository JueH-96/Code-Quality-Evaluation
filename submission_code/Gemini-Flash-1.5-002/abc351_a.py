# YOUR CODE HERE
A = list(map(int, input().split()))
B = list(map(int, input().split()))

takahashi_score = sum(A)
aoki_score = sum(B)

print(takahashi_score - aoki_score + 1)