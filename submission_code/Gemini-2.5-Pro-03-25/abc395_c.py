# YOUR CODE HERE
import sys
from collections import defaultdict

def solve():
    # Read N from standard input
    # N is the length of the sequence A
    N = int(sys.stdin.readline())
    
    # Constraints state N >= 1. If N=0 were possible, reading A might be an issue.
    # But with N >= 1, this is safe.
    
    # Read the sequence A as a list of integers from standard input
    A = list(map(int, sys.stdin.readline().split()))

    # Use a defaultdict to store a list of 0-based indices for each distinct value in A.
    # The key will be the value from A, and the value will be a list of indices
    # where this value appears.
    value_indices = defaultdict(list)
    for k in range(N):
        # Append the current index k to the list associated with the value A[k]
        value_indices[A[k]].append(k) 

    # Identify the set S of values that appear at least twice in the original sequence A.
    # A value x is in S if its list of indices in value_indices has length >= 2.
    S = set()
    for x in value_indices:
        # Check if the value x appears 2 or more times
        if len(value_indices[x]) >= 2:
            # If yes, add it to the set S
            S.add(x)

    # Check if the set S is empty.
    # If S is empty, it means no value appears multiple times in the original sequence A.
    # The problem requires finding a subarray that contains a repeated value 'x', AND 
    # this value 'x' must itself occur multiple times in the original sequence A.
    # If S is empty, the second part of this condition ('x' occurring multiple times in A) 
    # can never be satisfied. Therefore, no such subarray exists.
    if not S:
        # Print -1 as required and exit the function.
        print("-1")
        return

    # Initialize the minimum length found so far. 
    # We set it to N + 1 initially, which is guaranteed to be larger than any possible 
    # valid subarray length (the maximum possible length is N).
    min_len = N + 1 

    # Iterate through each value x present in the set S.
    # These are the values that appear multiple times in the original sequence A.
    for x in S:
        # Get the list of 0-based indices where the value x appears in A.
        indices = value_indices[x]
        
        # The problem asks for the shortest subarray satisfying the conditions.
        # A subarray A[i..j] satisfies the conditions if:
        # 1. It contains some value y at least twice.
        # 2. This value y is in S (i.e., y appears multiple times in A).
        #
        # Consider a value x from S. We want to find the shortest subarray containing x twice.
        # Such a subarray must span between two occurrences of x.
        # The shortest such subarrays are those defined by *adjacent* occurrences of x.
        # If x appears at indices k_1, k_2, ..., k_m (where k_1 < k_2 < ... < k_m),
        # the shortest subarrays containing x twice are of the form A[k_p ... k_{p+1}].
        # The length of such a subarray is k_{p+1} - k_p + 1.
        #
        # We iterate through all pairs of adjacent indices for the current value x.
        for i in range(len(indices) - 1):
            # indices[i] and indices[i+1] are the 0-based indices of two adjacent occurrences of x.
            # The subarray A[indices[i]...indices[i+1]] starts with x and ends with x.
            # Thus, it contains x at least twice.
            # Since x is in S, this subarray satisfies both problem conditions.
            
            # Calculate the length of this subarray. 
            # For 0-based indices p and q (where p <= q), the length is q - p + 1.
            length = indices[i+1] - indices[i] + 1
            
            # Update the overall minimum length found so far if the current subarray is shorter.
            min_len = min(min_len, length)

    # After iterating through all values x in S and considering the lengths derived from their
    # adjacent occurrences, min_len will hold the length of the shortest subarray that
    # satisfies the problem conditions.
    # Since S was non-empty, we are guaranteed to have found at least one valid subarray,
    # and min_len will have been updated to a value less than or equal to N.
    # The smallest possible length is 2 (this occurs if x appears at consecutive positions,
    # e.g., in A = [..., x, x, ...]).
    print(min_len)

# Call the solve function to execute the main logic of the program.
solve()