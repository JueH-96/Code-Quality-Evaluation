def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Handle the trivial case
    if N == 1:
        print(0)
        return

    # Sum of all elements
    S = sum(A)
    # base value x and remainder r
    x, r = divmod(S, N)

    # Sort array
    A.sort()

    # Construct target array in sorted order:
    # first (N-r) elements will be x, followed by r elements of (x+1)
    # We'll calculate the sum of absolute differences directly.
    # T[i] = x for i < (N-r), T[i] = x+1 for i >= (N-r)
    # Compare with sorted A to minimize sum of absolute differences.
    # Then the answer = half of that sum, guaranteed to be integer.
    min_diff_sum = 0

    # The first (N-r) elements should match x
    for i in range(N - r):
        min_diff_sum += abs(A[i] - x)
    # The last r elements should match x+1
    for i in range(N - r, N):
        min_diff_sum += abs(A[i] - (x + 1))

    # Each operation can shift 1 unit from any element to any other element,
    # reducing total difference by 2. So final cost is half of min_diff_sum.
    print(min_diff_sum // 2)

# Do not forget to call main()
if __name__ == "__main__":
    main()