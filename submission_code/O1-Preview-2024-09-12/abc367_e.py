# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))
    N = int(N)
    K = int(K)
    X = [x - 1 for x in X]  # Convert to 0-based indexing

    P = X  # The permutation mapping

    visited = [False] * N
    ans = [0] * N  # Initialize the answer list
    for idx in range(N):
        if not visited[idx]:
            # Build the cycle starting from idx
            cycle = []
            i = idx
            while not visited[i]:
                visited[i] = True
                cycle.append(i)
                i = P[i]

            L = len(cycle)
            k_mod = K % L

            for pos in range(L):
                old_idx = cycle[pos]
                new_pos = (pos + k_mod) % L
                new_idx = cycle[new_pos]
                ans[new_idx] = A[old_idx]

    print(' '.join(map(str, ans)))

threading.Thread(target=main).start()