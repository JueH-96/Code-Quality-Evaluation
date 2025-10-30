# YOUR CODE HERE
N, M, C, K = map(int, input().split())
A = list(map(int, input().split()))

result = 0
for k in range(K):
    result += min((C*k + a) % M for a in A)

print(result)