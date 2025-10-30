import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    MOD = 10**8

    # Part 1: Calculate (N-1) * sum_of_all_A_elements
    # Each A_k appears (N-1) times in the sum Sum_{i<j} (A_i + A_j).
    sum_of_all_A_elements = sum(A) 
    
    # This is Sum_{i<j} (A_i + A_j)
    term1 = (N - 1) * sum_of_all_A_elements

    # Part 2: Calculate MOD * C
    # C is the count of pairs (A_i, A_j) with i<j such that A_i + A_j >= MOD.
    # We sort array A to efficiently count these pairs.
    A.sort() # Sorts A in-place. This is O(N log N).

    count_pairs_ge_MOD = 0
    # Use the two-pointer technique on the sorted array A. This part is O(N).
    left = 0
    right = N - 1
    while left < right:
        if A[left] + A[right] >= MOD:
            # If A[left] + A[right] >= MOD:
            # Then A[right] paired with A[left], A[left+1], ..., A[right-1] 
            # will also satisfy sum >= MOD (because A[k] >= A[left] for k > left).
            # There are (right - 1) - left + 1 = right - left such pairs.
            count_pairs_ge_MOD += (right - left)
            right -= 1 # A[right] has been processed with all elements to its left.
        else: # A[left] + A[right] < MOD
            # If A[left] + A[right] < MOD:
            # Then A[left] is too small. Even when paired with A[right] (the largest 
            # element A[left] can be paired with from its right), the sum is less than MOD.
            # So, A[left] cannot form a pair (A[left], A[k]) for left < k <= right
            # whose sum is >= MOD (because A[k] <= A[right] for k < right).
            # Thus, we move to check the next element A[left+1].
            left += 1
            
    term2 = MOD * count_pairs_ge_MOD
    
    # The final answer is term1 - term2
    final_answer = term1 - term2
    
    sys.stdout.write(str(final_answer) + "
")

if __name__ == '__main__':
    main()