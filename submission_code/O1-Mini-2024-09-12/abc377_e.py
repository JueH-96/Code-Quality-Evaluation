# YOUR CODE HERE
import sys

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    P = [p - 1 for p in P]  # zero-based indexing

    visited = [False] * N
    result = [0] * N

    for i in range(N):
        if not visited[i]:
            # Find the cycle starting at i
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = P[current]
            cycle_length = len(cycle)
            shift = K % cycle_length
            for idx, val in enumerate(cycle):
                new_idx = (idx + shift) % cycle_length
                result[val] = cycle[new_idx] + 1  # back to 1-based

    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()