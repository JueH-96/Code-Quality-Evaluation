def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    degrees = [0] * (N + 1)
    for _ in range(N - 1):
        u = int(input[idx])
        v = int(input[idx + 1])
        idx += 2
        degrees[u] += 1
        degrees[v] += 1

    # Compute L_final
    L_final = 0
    for i in range(1, N + 1):
        if degrees[i] == 1:
            L_final += 1

    # Compute M
    M = (N - L_final + 2) // 3

    # Compute S
    S = L_final + 2 * (M - 1)

    # Collect non-leaf nodes' degrees
    non_leaf_degrees = []
    for i in range(1, N + 1):
        if degrees[i] >= 2:
            non_leaf_degrees.append(degrees[i])

    # Sort and print
    non_leaf_degrees.sort()
    print(' '.join(map(str, non_leaf_degrees)))

if __name__ == '__main__':
    main()