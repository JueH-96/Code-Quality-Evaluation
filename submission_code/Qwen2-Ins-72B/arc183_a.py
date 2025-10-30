import numpy as np
N, K = map(int, input().split())

ans = np.repeat(np.arange(1, N+1), K)
ans[K-1] = 1
for i in range(1, N):
    if K % 2 == 0:
        ans[K*i-1:K*i] = ans[K*i-1:K*i][::-1]
    else:
        ans[K*i-1:K*i+1] = ans[K*i-1:K*i+1][::-1]

print(*ans)