import sys

# Read input
N = int(sys.stdin.readline().strip())
q_r = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]
Q = int(sys.stdin.readline().strip())
t_d = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(Q)]

# Solve the problem
for t, d in t_d:
    q, r = q_r[t-1]
    next_day = d + (q - (d - r) % q) % q
    print(next_day)