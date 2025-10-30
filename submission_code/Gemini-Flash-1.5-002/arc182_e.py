# YOUR CODE HERE
N, M, C, K = map(int, input().split())
A = list(map(int, input().split()))

ans = 0
for k in range(K):
    min_val = float('inf')
    for a in A:
        min_val = min(min_val, (C * k + a) % M)
    ans += min_val

print(ans)