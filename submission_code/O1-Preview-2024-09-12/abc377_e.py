# YOUR CODE HERE
import sys
import threading
def main():
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    N = int(N)
    K = int(K)
    P = [x - 1 for x in P]  # Adjust to zero-based indices
    res = [0]*N
    visited = [False]*N
    for i in range(N):
        if not visited[i]:
            # Build cycle
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = P[j]
            L = len(cycle)
            M = pow(2, K, L)
            for idx in range(L):
                new_idx = (idx + M)%L
                res[cycle[new_idx]] = P[cycle[idx]]
    # Adjust res back to one-based indices
    res = [x + 1 for x in res]
    print(' '.join(map(str, res)))

threading.Thread(target=main).start()