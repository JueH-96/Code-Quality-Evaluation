# YOUR CODE HERE
import sys
import bisect

# Read N
N = int(sys.stdin.readline())

# Read A
# A is already sorted in ascending order as per constraints.
A = list(map(int, sys.stdin.readline().split()))

total_kagamimochi = 0

# Iterate through each mochi in the list A.
# We consider each mochi A[i] as the potential top mochi in a kagamimochi stack.
for i in range(N):
    top_size = A[i]

    # The rule for stacking is: mochi A (size a) on top of mochi B (size b)
    # is allowed if a <= b / 2.
    # For the current top mochi A[i] with size top_size, we need a bottom mochi A[j]
    # with size A[j] such that top_size <= A[j] / 2.
    # This inequality is equivalent to top_size * 2 <= A[j].

    # We need to find the number of mochi A[j] in the list such that A[j] is
    # greater than or equal to `top_size * 2`.
    target_bottom_size = top_size * 2

    # Since the list A is sorted, all elements greater than or equal to
    # `target_bottom_size` will appear contiguously at the end of the list,
    # starting from a certain index.
    # bisect_left(A, x) finds the index k such that A[k-1] < x <= A[k].
    # This means A[k] is the first element in A that is >= x.
    # All elements from index k to N-1 are >= x.
    first_bottom_idx = bisect.bisect_left(A, target_bottom_size)

    # The number of such elements (potential bottom mochi) is N - first_bottom_idx.
    # These are the indices j from first_bottom_idx to N-1.
    num_valid_bottom = N - first_bottom_idx

    # The problem states we choose two different mochi, implying i != j.
    # The condition A[j] >= A[i] * 2 already guarantees i != j because A[i] >= 1.
    # If i == j were possible, it would require A[i] >= A[i] * 2, which simplifies
    # to A[i] <= 0. However, the constraint is 1 <= A_i <= 10^9, so A[i] is always >= 1.
    # Thus, A[i] >= A[i] * 2 is never true, and the index i itself will never be
    # included in the set of indices j found by bisect_left(A, target_bottom_size).
    # So, the count `N - first_bottom_idx` correctly represents the number of distinct
    # mochi A[j] (where j != i) that can be the bottom layer for the current top mochi A[i].

    # Add the number of valid bottom mochi found for the current top mochi to the total count.
    total_kagamimochi += num_valid_bottom

# Print the final total number of different kinds of kagamimochi that can be made.
print(total_kagamimochi)