import sys
from collections import deque

def main():
    N, K = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))

    X = [x-1 for x in X]
    A = [a-1 for a in A]

    visited = [0]*N
    cycle = deque()
    cycle.append(0)
    visited[0] = 1

    while visited[X[cycle[-1]]] == 0:
        cycle.append(X[cycle[-1]])
        visited[cycle[-1]] = 1

    start = cycle.index(X[cycle[-1]])
    length = len(cycle) - start

    if K < start:
        print(' '.join(str(A[cycle[K]]+1) for _ in range(N)))
    else:
        K -= start
        K %= length
        print(' '.join(str(A[cycle[start+K]]+1) for _ in range(N)))

if __name__ == "__main__":
    main()