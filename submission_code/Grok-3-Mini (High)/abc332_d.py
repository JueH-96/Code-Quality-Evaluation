import sys
import itertools

def inversion_count(perm):
    cnt = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                cnt += 1
    return cnt

# Read input
data = sys.stdin.read().split()
idx = 0
H = int(data[idx])
idx += 1
W = int(data[idx])
idx += 1

A = []
for _ in range(H):
    row = [int(data[idx + k]) for k in range(W)]
    A.append(row)
    idx += W

B = []
for _ in range(H):
    row = [int(data[idx + k]) for k in range(W)]
    B.append(row)
    idx += W

# Find minimum operations
min_ops = float('inf')
for sigma in itertools.permutations(range(H)):
    for tau in itertools.permutations(range(W)):
        match = True
        for i in range(H):
            for j in range(W):
                if A[sigma[i]][tau[j]] != B[i][j]:
                    match = False
                    break
            if not match:
                break
        if match:
            inv_sig = inversion_count(sigma)
            inv_tau = inversion_count(tau)
            cost = inv_sig + inv_tau
            if cost < min_ops:
                min_ops = cost

# Output the result
if min_ops == float('inf'):
    print(-1)
else:
    print(min_ops)