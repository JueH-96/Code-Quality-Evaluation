# YOUR CODE HERE
import sys
from collections import defaultdict

def solve():
    """
    Solves the problem based on the given specifications.
    Reads input N and sequence A, then calculates the number of triples (i, j, k)
    such that 1 <= i < j < k <= N, A[i] = A[k], and A[i] != A[j].
    The indices i, j, k in the problem statement are 1-based.
    The implementation uses 0-based indexing internally for array access,
    but the logic correctly reflects the problem conditions.
    """
    N = int(sys.stdin.readline())
    # Read the sequence A. Using list(map(...)) is standard and efficient enough.
    A = list(map(int, sys.stdin.readline().split()))

    # Use defaultdict to store lists of indices for each value in A.
    # Keys are values from A, values are lists of 0-based indices where the key appears.
    value_indices = defaultdict(list)
    for i in range(N):
        value_indices[A[i]].append(i) # Store 0-based index i

    total_count = 0 # Initialize the total count of valid triples

    # Iterate through each distinct value present in the sequence A
    for val in value_indices:
        # Get the list of 0-based indices where 'val' appears
        indices = value_indices[val]
        # 'm' is the number of times 'val' appears in A
        m = len(indices)
        
        # To form a pair (i, k) such that A[i] = A[k] = val and i < k,
        # we need at least two occurrences of 'val'.
        # If m < 2, this value cannot contribute to any valid triple.
        if m < 2:
            continue 

        # The derived formula for the count contributed by value 'val' is:
        # Sum_{k=0}^{m-1} (2k - m + 1) * p'_k - C(m+1, 3)
        # where p'_k is the k-th 0-based index of 'val' (indices[k]).

        # Calculate the first part: Sum_{k=0}^{m-1} (2k - m + 1) * p'_k
        current_sum_term = 0
        for k in range(m):
            p_k = indices[k] # This is p'_k, the 0-based index
            # Calculate the coefficient for p'_k
            coefficient = (2 * k) - m + 1
            # Add the term to the sum
            current_sum_term += coefficient * p_k

        # Calculate the second part: Binomial coefficient C(m+1, 3)
        # C(n, k) = n! / (k! * (n-k)!)
        # C(m+1, 3) = (m+1) * m * (m-1) / (3 * 2 * 1)
        # This is well-defined and potentially non-zero only if m+1 >= 3, i.e., m >= 2.
        # The 'if m < 2: continue' check ensures m >= 2 here.
        # Use integer division // to ensure the result is an integer.
        # Python's arbitrary precision integers handle large intermediate products.
        binom_m_plus_1_3 = (m + 1) * m * (m - 1) // 6
        
        # The total count for triples where A[i] = A[k] = val is the sum term minus C(m+1, 3).
        # Add this contribution to the overall total count.
        total_count += current_sum_term - binom_m_plus_1_3

    # Print the final computed total count.
    print(total_count)

# Call the solve function to execute the logic.
solve()