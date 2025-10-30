import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    P = [0] + P  # Convert to 1-based indexing

    visited = [False] * (N + 1)
    cycles = []

    # Find all cycles in the permutation
    for i in range(1, N + 1):
        if not visited[i]:
            cycle = []
            current = i
            while True:
                if visited[current]:
                    break
                visited[current] = True
                cycle.append(current)
                current = P[current]
            cycles.append(cycle)

    res = [0] * (N + 1)

    # Process each cycle to compute the result after K operations
    for cycle in cycles:
        L = len(cycle)
        if L == 0:
            continue
        # Calculate the effective steps modulo the cycle length
        m = pow(2, K, L)
        for idx in range(L):
            original = cycle[idx]
            new_idx = (idx + m) % L
            res[original] = cycle[new_idx]

    # Output the result
    print(' '.join(map(str, res[1:N+1])))

if __name__ == '__main__':
    main()