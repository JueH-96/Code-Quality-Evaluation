import sys

def can_make(K, N, A):
    """
    Checks if it's possible to make K kagamimochi.
    Assumes A is sorted and 0-indexed.
    K is the number of pairs.
    N is the total number of mochi.
    A is the list of mochi sizes.
    """
    # If K == 0, it's always possible to make 0 kagamimochi.
    if K == 0:
        return True

    # We need 2*K distinct mochi. The maximum possible K is N // 2.
    # The binary search range [0, N // 2] ensures 2*K <= N.
    # The problem constraints state N >= 2, so N // 2 >= 1, and the range is valid.

    # To make K pairs (a_i, b_i) with a_i <= b_i / 2 using 2K distinct mochi,
    # we should choose the K smallest mochi from the original list A as potential top pieces,
    # and the K largest mochi from the original list A as potential bottom pieces.
    # These are A[0], ..., A[K-1] and A[N-K], ..., A[N-1] respectively (0-indexed).
    # These 2K mochi are distinct because the indices {0, ..., K-1} and {N-K, ..., N-1}
    # are disjoint if K-1 < N-K, i.e., 2*K <= N, which is ensured by the binary search range [0, N//2].

    # The condition for these K smallest mochi A[0...K-1] and K largest mochi A[N-K...N-1]
    # to form K pairs such that the i-th smallest top piece can be paired with the i-th
    # smallest bottom piece among the chosen sets is:
    # 2 * (i-th smallest top piece) <= (i-th smallest bottom piece)
    # The i-th smallest among A[0...K-1] is A[i-1] (1-based index).
    # The i-th smallest among A[N-K...N-1] is A[N-K+i-1] (1-based index).
    # Using 0-based indexing for A, this means checking:
    # 2 * A[i] <= A[N-K+i] for i = 0, ..., K-1.

    for i in range(K):
        # A[i] is the (i+1)-th mochi overall (smallest available as top).
        # A[N-K+i] is the (N-K+i+1)-th mochi overall (smallest available as bottom for this scheme).
        if 2 * A[i] > A[N - K + i]:
            return False

    return True

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # A is guaranteed to be sorted in ascending order.

    # Binary search for the maximum possible K.
    # K represents the number of pairs. We need 2*K distinct mochi.
    # So K can range from 0 up to N // 2.
    low = 0
    high = N // 2
    ans = 0 # Minimum possible pairs is 0

    while low <= high:
        mid = (low + high) // 2
        if can_make(mid, N, A):
            # If we can make 'mid' pairs using the strategy,
            # then 'mid' is a possible answer. We try if a larger K is possible.
            ans = mid
            low = mid + 1
        else:
            # If we cannot make 'mid' pairs using the strategy,
            # then 'mid' is not possible, and neither are larger values of K.
            # We need to try fewer pairs.
            high = mid - 1

    print(ans)

# Entry point for the script
if __name__ == "__main__":
    solve()