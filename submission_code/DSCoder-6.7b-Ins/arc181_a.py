import sys

def solve():
    T = int(sys.stdin.readline().strip())
    for _ in range(T):
        N = int(sys.stdin.readline().strip())
        P = list(map(int, sys.stdin.readline().strip().split()))
        operations = 0
        for i in range(N):
            while P[i] != i+1:
                operations += 1
                if P[i] > i+1:
                    P[i:P[i]] = sorted(P[i:P[i]])
                else:
                    P[i+1:] = sorted(P[i+1:])
        print(operations)

solve()