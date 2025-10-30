import sys

def main():
    N = int(sys.stdin.readline())
    degrees = [0] * (N + 1)  # 1-based indexing

    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        degrees[u] += 1
        degrees[v] += 1

    # Calculate K: number of leaves (degree 1 nodes)
    K = sum(1 for i in range(1, N + 1) if degrees[i] == 1)

    # Calculate M (number of stars)
    M = (N - K + 2) // 3

    # Calculate T2 (number of nodes with degree 2)
    T2 = sum(1 for i in range(1, N + 1) if degrees[i] == 2)

    # Calculate X (number of degree-2 centers)
    X = T2 - 2 * (M - 1)

    # Collect degrees of nodes with degree >= 3
    res = []
    for i in range(1, N + 1):
        if degrees[i] >= 3:
            res.append(degrees[i])

    # Add X nodes of degree 2
    res += [2] * X

    # Sort and print
    res.sort()
    print(' '.join(map(str, res)))

if __name__ == '__main__':
    main()