# YOUR CODE HERE
N, M, C, K = map(int, input().split())
A = list(map(int, input().split()))

min_values = [float('inf')] * M
for a in A:
    for k in range(M):
        min_values[k] = min(min_values[k], (C * k + a) % M)

result = 0
for k in range(K):
    result += min_values[k % M]

print(result)