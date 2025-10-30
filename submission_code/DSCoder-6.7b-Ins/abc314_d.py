import sys

N = int(sys.stdin.readline().strip())
S = list(sys.stdin.readline().strip())
Q = int(sys.stdin.readline().strip())

for _ in range(Q):
    t, x, c = sys.stdin.readline().strip().split()
    t, x = int(t), int(x)
    if t == 1:
        S[x-1] = c
    elif t == 2:
        S = [s.upper() if s.islower() else s for s in S]
    else:
        S = [s.lower() if s.isupper() else s for s in S]

print(''.join(S))