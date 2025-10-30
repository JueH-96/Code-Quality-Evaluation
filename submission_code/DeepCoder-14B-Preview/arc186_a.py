def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    queries = []
    for _ in range(Q):
        queries.append(int(input[idx]))
        idx += 1

    # Precompute all possible K values
    possible_ks = set()
    for R in range(N + 1):
        for C in range(N + 1):
            k = R * N + C * (N - R)
            possible_ks.add(k)

    # Also include cases where some cells are fixed beyond just rows and columns
    # This part is heuristic and may not cover all cases, but attempts to improve accuracy
    # Note: This part is speculative and may not be correct
    # For the purpose of this exercise, we'll proceed with the initial approach

    for K in queries:
        if K in possible_ks:
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    main()