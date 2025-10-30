# YOUR CODE HERE
import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    M = 10**8 # The modulus for f(x, y) = (x + y) % M

    # Calculate the sum of A_i + A_j for all pairs (i, j) with i < j, without modulo.
    # This sum can be rewritten as (N-1) * sum(A_k) for k from 1 to N.
    # Each A_k appears (N-1) times in the overall sum:
    # A_k is paired with A_1, ..., A_{k-1} (k-1 times)
    # A_k is paired with A_{k+1}, ..., A_N (N-k times)
    # Total times A_k appears = (k-1) + (N-k) = N-1.
    total_sum_A = sum(A)
    S_total_unmod = (N - 1) * total_sum_A

    # Calculate count_overflow: the number of pairs (i, j) with i < j
    # such that A_i + A_j >= M.
    # For each such pair, the actual f(A_i, A_j) is (A_i + A_j - M).
    # This means we subtract M for each such pair from S_total_unmod.

    # Use a two-pointer approach on the sorted array A to efficiently count `count_overflow`.
    A_sorted = sorted(A)
    
    count_overflow = 0
    left = 0
    right = N - 1

    # The loop continues as long as the left pointer is to the left of the right pointer.
    # This ensures we consider unique pairs (i, j) where i < j (after sorting).
    while left < right:
        current_pair_sum = A_sorted[left] + A_sorted[right]
        
        if current_pair_sum >= M:
            # If A_sorted[left] + A_sorted[right] is greater than or equal to M,
            # it means A_sorted[right] is large enough to cause an overflow with A_sorted[left].
            # Since A_sorted is sorted in non-decreasing order, any element A_sorted[k]
            # where `left <= k < right` will satisfy A_sorted[k] >= A_sorted[left].
            # Therefore, A_sorted[k] + A_sorted[right] will also be >= A_sorted[left] + A_sorted[right] (which is >= M).
            # This means A_sorted[right] forms an overflow pair with A_sorted[left], A_sorted[left+1], ..., A_sorted[right-1].
            # The number of such elements is (right - left).
            # We add this count to count_overflow.
            count_overflow += (right - left)
            
            # Now that we have counted all pairs involving A_sorted[right] (that cause overflow with elements to its left),
            # we move the right pointer inward to consider potentially smaller sums.
            right -= 1
        else:
            # If A_sorted[left] + A_sorted[right] is less than M,
            # it means A_sorted[left] is too small to cause an overflow with A_sorted[right].
            # Since A_sorted[right] is the largest element currently available from the right side,
            # and A_sorted is sorted, A_sorted[left] will also not form an overflow pair with any
            # element A_sorted[k] where `k` is between `left+1` and `right`.
            # Thus, we need to try a larger element for the left side.
            # We move the left pointer inward.
            left += 1
            
    # The final answer is the sum of all A_i + A_j pairs minus M for each pair that overflowed.
    final_answer = S_total_unmod - M * count_overflow
    
    print(final_answer)

solve()