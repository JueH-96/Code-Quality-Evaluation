import sys

# Read input
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

# Initialize total count of different kagamimochi
total_kagamimochi = 0

# Use two pointers. The outer loop iterates through each mochi A[j] as the
# potential bottom mochi. The inner mechanism finds the number of potential
# top mochi A[i] (with i < N) such that A[i] <= A[j] / 2.
# Since A is sorted, if A[k] <= A[j] / 2, then all A[p] with p < k also satisfy
# A[p] <= A[k] <= A[j] / 2. So the potential top mochi are A[0], A[1], ..., A[k']
# for some k'. The number of such mochi is k'+1.
# We can use a two-pointer approach where the right pointer `j` iterates from 0 to N-1,
# and the left pointer `i` keeps track of the count of elements A[0]...A[i-1] that are
# less than or equal to A[j] / 2.0. As `j` increases, A[j] / 2.0 is non-decreasing,
# so the pointer `i` only needs to advance forward.

i = 0 # Pointer for finding the count of potential top mochi A[0]...A[i-1]

# Iterate through each mochi A[j] as the potential bottom mochi
# The index j ranges from 0 to N-1
for j in range(N):
    bottom_size = A[j]

    # The condition for a top mochi A[k] to be placed on A[j] is A[k] <= A[j] / 2
    # Calculate the maximum allowed size for the top mochi.
    # Use float division to handle cases like A[j] = 7, where A[j]/2 = 3.5
    target_top_size = bottom_size / 2.0

    # Advance pointer i while A[i] is a valid size for a top mochi on A[j]
    # This finds the count of elements A[0]...A[i-1] such that A[k] <= target_top_size.
    # The pointer `i` will stop at the first index where A[i] > target_top_size
    # (or at N if all elements are <= target_top_size).
    # The value of `i` at the end of this while loop is the number of elements
    # A[0]...A[i-1] that satisfy the condition A[k] <= target_top_size.
    # This is equivalent to bisect_right(A, target_top_size).
    # Since j is strictly increasing (from 0 to N-1), A[j] is non-decreasing,
    # target_top_size is non-decreasing. Thus, the necessary value of `i`
    # for A[j+1] will be >= the necessary value of `i` for A[j].
    # So we can continue incrementing `i` from its previous position.
    while i < N and A[i] <= target_top_size:
        i += 1

    # At this point, `i` is the number of mochi A[0]...A[i-1] that can be a top mochi
    # for the current bottom mochi A[j]. There are `i` such mochi.
    # For each choice A[k] with 0 <= k < i, the pair (k, j) is a valid kagamimochi (A[k] on A[j]).
    # We need k != j.
    # The condition A[k] <= A[j]/2.0 implies A[k] < A[j] for A[j] >= 1.
    # Since A is sorted, A[k] < A[j] implies k < j.
    # The `while` loop finds all indices k from 0 up to `i-1` such that A[k] <= A[j]/2.0.
    # All these indices k are guaranteed to be less than j (k < j), since k < i.
    # If i <= j, then k < i <= j, implies k < j.
    # If i > j, it would imply A[j] <= A[j]/2.0 which is impossible for A[j] >= 1.
    # Therefore, all k in the range 0 <= k < i are strictly less than j, ensuring k != j.
    # So, for the current bottom mochi A[j], there are exactly `i` possible top mochi.

    total_kagamimochi += i

# Print the total count
print(total_kagamimochi)