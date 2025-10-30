import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())
    # Read the list of mochi sizes A
    # The input guarantees A is already sorted in ascending order.
    A = list(map(int, sys.stdin.readline().split()))

    # We want to find the maximum integer K such that we can choose 2K mochi
    # from the N mochi and form K pairs (a, b) satisfying a <= b / 2.

    # We can use a two-pointer approach on the sorted array A.
    # We try to pair the smallest available mochi as the top layer (a)
    # with a suitable larger available mochi as the bottom layer (b).
    # To maximize the number of pairs, a greedy strategy is effective.
    # Consider partitioning the sorted array A into two conceptual parts:
    # The first part A[0] to A[N//2 - 1] as potential top mochi,
    # The second part A[N//2] to A[N - 1] as potential bottom mochi.
    # Using two pointers, we iterate through the potential tops from the first part
    # and try to match them with the smallest possible potential bottom
    # from the second part that satisfies the condition.

    # i is the pointer for the potential top mochi (from the first part)
    # It starts at the first index 0.
    i = 0

    # j is the pointer for the potential bottom mochi (from the second part)
    # It starts at the first index of the second part, N // 2.
    j = N // 2

    # count stores the number of kagamimochi pairs we can form.
    count = 0

    # We continue pairing as long as we have potential top mochi available
    # in the first part and potential bottom mochi available in the second part.
    # The indices for the first part are 0 to N // 2 - 1.
    # The indices for the second part are N // 2 to N - 1.
    # The range of i is [0, N // 2), and the range of j is [N // 2, N).
    # These two ranges are disjoint, ensuring the selected 2K mochi are distinct.
    while i < N // 2 and j < N:
        # Check if A[i] (potential top) can be placed on A[j] (potential bottom).
        # The condition is A[i] <= A[j] / 2, or equivalently, A[i] * 2 <= A[j].
        # Using A[i] * 2 <= A[j] avoids floating-point arithmetic and potential precision issues.
        if A[i] * 2 <= A[j]:
            # A[i] can be paired with A[j]. We form this pair.
            # Increment the count of successful pairs.
            count += 1
            # Move to the next potential top mochi.
            i += 1
            # Move to the next potential bottom mochi.
            j += 1
        else:
            # A[j] is too small to be a bottom for A[i] (A[i] * 2 > A[j]).
            # Since the array A is sorted in ascending order, any subsequent
            # potential top mochi A[i'] with i' > i will have A[i'] >= A[i].
            # Therefore, A[j] will also be too small to be a bottom for A[i']
            # because A[i'] * 2 >= A[i] * 2 > A[j].
            # This means A[j] cannot be used as a bottom for the current A[i]
            # or any subsequent potential top from the first part.
            # We must discard A[j] as a potential bottom for the remaining tops
            # and move to the next larger potential bottom mochi.
            j += 1

    # The loop finishes when we run out of potential tops in the first part (i reaches N // 2)
    # or when we run out of potential bottoms in the second part (j reaches N).
    # The value of count at the end is the maximum number of kagamimochi pairs
    # that can be formed using this strategy. This strategy finds the maximum K.

    # Print the result.
    print(count)

solve()