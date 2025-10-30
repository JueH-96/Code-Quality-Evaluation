# YOUR CODE HERE
N, X = map(int, input().split())
S = list(map(int, input().split()))

total_score = 0
for i in range(N):
    if S[i] <= X:
        total_score += S[i]

print(total_score)