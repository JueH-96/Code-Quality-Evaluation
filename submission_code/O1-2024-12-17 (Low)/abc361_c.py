def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # We want to keep N - K elements so that (max - min) among the chosen is minimized.
    # Because we can choose any subset of size N-K (no adjacency constraint),
    # we simply need to pick the N-K consecutive elements in sorted order that yield the smallest difference.
    
    M = N - K  # size of the subset to keep
    A.sort()

    # Initialize result with a large value.
    best_diff = float('inf')

    # Slide over all possible consecutive segments of length M in the sorted array.
    for i in range(N - M + 1):
        diff = A[i + M - 1] - A[i]
        if diff < best_diff:
            best_diff = diff

    print(best_diff)

# Do not forget to call the main function
if __name__ == "__main__":
    main()