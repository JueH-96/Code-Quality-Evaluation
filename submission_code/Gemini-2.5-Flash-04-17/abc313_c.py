import sys

def solve():
    # Read the number of elements N
    N = int(sys.stdin.readline())

    # Read the sequence A
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the sum of elements in A
    # Use Python's arbitrary precision integers for sum, as A_i can be large and N up to 2e5.
    S = sum(A)

    # If the difference between min and max is at most one, the elements can only
    # take two values: q and q+1, where q is the floor of the average.
    # S = N * q + r, where q = S // N and r = S % N, with 0 <= r < N.
    # This implies that exactly r elements must be equal to q+1 and N-r elements must be equal to q.
    q = S // N
    r = S % N

    target_low = q
    target_high = q + 1

    # Number of elements that should ideally be target_high (q+1) is r.
    # Number of elements that should ideally be target_low (q) is N-r.
    num_high = r
    num_low = N - r # This is N - S % N

    # To minimize the number of operations (total units moved), we should make
    # the smallest N-r elements of the initial array become target_low, and
    # the largest r elements of the initial array become target_high.
    # Sorting the array A helps in identifying the smallest and largest elements.
    A.sort()

    # The minimum number of operations is the total amount of "stuff" that needs
    # to be added to elements to bring them up to their target values. This is the
    # total deficit: sum of max(0, target_value - A[i]) for each element.
    # Alternatively, it's the total excess: sum of max(0, A[i] - target_value).
    # Both sums are equal. We calculate the total deficit.

    total_ops = 0

    # The first num_low elements in the sorted array (indices 0 to num_low - 1)
    # should ideally become target_low (q).
    # We calculate the deficit for these elements.
    for i in range(num_low):
        # The required increase for A[i] is target_low - A[i].
        # If A[i] is already >= target_low, no increase is needed (max(0, ...) handles this).
        # Since A is sorted, A[i] for i < num_low are the smallest elements.
        # target_low = floor(S/N). It's possible some of the smallest A[i] are
        # greater than target_low, especially if the initial numbers are very spread out.
        # Using max(0, target_low - A[i]) is correct.
        total_ops += max(0, target_low - A[i])

    # The remaining num_high elements in the sorted array (indices num_low to N - 1)
    # should ideally become target_high (q + 1).
    # We calculate the deficit for these elements.
    for i in range(num_low, N):
        # The required increase for A[i] is target_high - A[i].
        # If A[i] is already >= target_high, no increase is needed (max(0, ...) handles this).
        # Since A is sorted, A[i] for i >= num_low are the largest elements.
        # target_high = floor(S/N) + 1. It's possible some of the largest A[i] are
        # less than target_high.
        # Using max(0, target_high - A[i]) is correct.
        total_ops += max(0, target_high - A[i])

    # Print the minimum number of operations
    print(total_ops)

solve()