import sys
from itertools import combinations
from functools import reduce
import operator

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Fast I/O
        readline = sys.stdin.readline

        # Read N and K
        N, K = map(int, readline().split())
        
        # Read the sequence A
        A = list(map(int, readline().split()))

        # The core idea is that C(N, K) <= 10^6 implies min(K, N - K) must be small.
        # We can iterate through combinations of size min(K, N - K) efficiently.
        
        # Determine if it's more efficient to iterate through combinations
        # of elements to take (K) or elements to leave out (N-K).
        # We choose the smaller of the two set sizes.
        if K > N / 2:
            # Dual approach: choose N-K elements to exclude.
            # max(XOR_sum(subset of size K)) = max(total_XOR ^ XOR_sum(subset of size N-K))
            k_to_choose = N - K
            is_dual = True
            # Pre-calculate the XOR sum of all elements in A.
            total_xor = reduce(operator.xor, A, 0)
        else:
            # Direct approach: choose K elements.
            k_to_choose = K
            is_dual = False

        max_xor_sum = 0
        
        # C(N, k_to_choose) is guaranteed to be <= 10^6.
        # The complexity of this loop is O(C(N, k_to_choose) * k_to_choose),
        # which is feasible under the problem constraints.
        for combo in combinations(A, k_to_choose):
            # Calculate the XOR sum of the current combination.
            # reduce with operator.xor is an efficient way to do this.
            subset_xor = reduce(operator.xor, combo, 0)
            
            if is_dual:
                # The value for the K chosen elements is total_xor ^ subset_xor
                current_xor_sum = total_xor ^ subset_xor
            else:
                # The value is simply the XOR sum of the K chosen elements.
                current_xor_sum = subset_xor
            
            # Update the maximum value found so far.
            if current_xor_sum > max_xor_sum:
                max_xor_sum = current_xor_sum
        
        # Print the final answer.
        print(max_xor_sum)

    except (IOError, ValueError):
        # This part handles potential I/O errors, e.g., empty input.
        # It's good practice but often not strictly necessary in competitive programming contests.
        pass

# Run the solution
solve()