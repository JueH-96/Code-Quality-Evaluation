import sys

def solve(N):
    V_N = int(str(N)*N)
    return V_N % 998244353

for line in sys.stdin:
    N = int(line.strip())
    print(solve(N))