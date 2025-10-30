import sys

N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()
Q = int(sys.stdin.readline().strip())

for _ in range(Q):
    c, d = sys.stdin.readline().strip().split()
    S = S.replace(c, d)

print(S)