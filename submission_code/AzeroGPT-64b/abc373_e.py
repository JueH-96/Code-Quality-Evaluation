import sys
readline = sys.stdin.readline
inf = 10 ** 18
N, M, K = map(int, readline().split())
A = list(map(int, readline().split()))
Z = sorted([(a, i) for i, a in enumerate(A)])
cand = [inf + 10] * N
for i in range(N - M, -1, -1):
  a, i = Z[i]
  b, _ = Z[i+1] if i<M-1 else (a+inf, 0)
  cand[i] = min(cand[i], max(0, b - a - M + 1))
  
for i, a in Z[M-1:]:
  b, _ = Z[i+1] if i<N-1 else (a+inf, 0)
  cand[i] = min(cand[i], max(0, b - a))

for i in range(N):
  c = cand[i]
  a = A[i]
  if c > K - a:
    c = -1
  print(c, end=' ')