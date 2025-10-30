import numpy as np

N, K = map(int, input().split())
AB = np.array([list(map(int, input().split())) for _ in range(N)])

dp = np.zeros((N, K+1), dtype=np.int64)
dp[:, 0] = AB[:, 1]

for k in range(1, K+1):
    dp[:, k] = np.max(dp[:, k-1] * AB[:, 0] + AB[:, 1], axis=1)

print(np.max(dp[:, K]))