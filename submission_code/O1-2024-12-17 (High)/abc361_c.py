def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # We need to pick exactly N-K elements so as to minimize (max - min) of the chosen subsequence.
    # Key insight: if we sort A, then picking any (N-K) elements whose values are as close as possible
    # will achieve the minimum difference. Because there is no adjacency requirement, just that the
    # chosen elements form a subsequence in the original order. As soon as a value range has at least
    # N-K elements, we can pick them in ascending index order from the original array (skipping
    # everything else).
    #
    # Therefore, the problem reduces to:
    #  1) Sort the entire array.
    #  2) Slide a window of size (N-K) over the sorted array to find the minimal difference
    #     between the first and last element of each window.
    #
    # The minimal such difference is our answer.

    A.sort()
    length = N - K  # number of elements to keep
    min_diff = 10**18

    # We'll examine each consecutive block of size (N-K) in the sorted array
    for i in range(K + 1):
        diff = A[i + length - 1] - A[i]
        if diff < min_diff:
            min_diff = diff

    print(min_diff)

# Do not forget to call main()
main()