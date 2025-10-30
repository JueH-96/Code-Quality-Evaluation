import sys

def solve():
    input = sys.stdin.read().split()
    T = int(input[0])
    cases = list(map(int, input[1:T+1]))
    for N in cases:
        if N == 1:
            print(20250126, 1)
        else:
            A = 2
            M = (1 << N) - 1
            print(A, M)

solve()