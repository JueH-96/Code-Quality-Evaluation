# YOUR CODE HERE
import sys
from collections import Counter

def solve():
    """
    Solves the problem by finding the minimum difference between the maximum and minimum elements 
    in a subsequence of length N-K formed by removing K elements from the original sequence A.
    The approach uses a two-pointer technique on the sorted unique values of A.
    """
    
    # Read N and K from the first line of input
    N, K = map(int, sys.stdin.readline().split())
    
    # Read the sequence A from the second line of input
    # Check if N > 0 before reading A to potentially handle edge cases, though constraints state N >= 2
    if N > 0:
      A = list(map(int, sys.stdin.readline().split()))
    else:
      # N=0 is not possible by constraints, but defensively handle.
      A = [] 

    # Calculate the required length of the resulting subsequence B
    M = N - K 

    # Constraints state 1 <= K < N. This implies N >= 2.
    # N - K >= N - (N-1) = 1. So M >= 1.
    # N - K <= N - 1. So M <= N-1.

    # Base case: If the resulting subsequence B needs only 1 element (M=1), 
    # the difference max(B) - min(B) is always 0, as max and min are the same element.
    if M == 1: 
        print(0)
        return

    # Use Counter to efficiently count occurrences of each number in A
    counts = Counter(A)
    
    # Get the unique values from A and sort them in ascending order
    unique_A = sorted(counts.keys())
    
    # U stores the number of unique values in A
    U = len(unique_A)
    
    # If there are no unique elements (only possible if N=0, which is ruled out by constraints)
    # This check is defensive. For N>=2, U >= 1.
    if U == 0:
        print(0) 
        return

    # Create a list `value_counts` where value_counts[i] is the count of unique_A[i] in A
    value_counts = []
    for val in unique_A:
        value_counts.append(counts[val])

    # Initialize `min_diff` to positive infinity. This will store the minimum difference found.
    min_diff = float('inf') 
    
    # Initialize `current_count` to 0. This tracks the total count of elements within the current sliding window [unique_A[i], unique_A[j-1]].
    current_count = 0 
    # Initialize `j` to 0. This is the right pointer for the window of unique values. 
    # It points to the index of the *next* unique value to potentially include in the window.
    j = 0 
    
    # Iterate through all unique values `unique_A[i]` using `i` as the left pointer of the window.
    # Each `unique_A[i]` is considered as a potential minimum value for the subsequence B.
    for i in range(U): 
        # When the left pointer `i` moves one step to the right (for i > 0), 
        # the value `unique_A[i-1]` is effectively removed from the left end of the window.
        # So, subtract its count from `current_count`.
        if i > 0:
            current_count -= value_counts[i-1]
        
        # Expand the window to the right using the pointer `j`.
        # Continue adding counts of `unique_A[j]` as long as `j` is within the bounds of `unique_A` 
        # and the `current_count` is less than the required number of elements M.
        while j < U and current_count < M:
             current_count += value_counts[j]
             # Move `j` to the right, pointing to the next unique value.
             j += 1 

        # After the while loop, the current window of values considered is [unique_A[i], unique_A[j-1]].
        # Check if this window contains at least M elements in total.
        if current_count >= M:
             # If yes, we have found a valid range of values. It's possible to form a subsequence B
             # of length M using only elements from A that fall within this range.
             # The difference between the maximum and minimum values in this range is `unique_A[j-1] - unique_A[i]`.
             # This difference is an upper bound on the actual `max(B) - min(B)` for any B formed from this range.
             # We update `min_diff` with the minimum difference found so far.
             min_diff = min(min_diff, unique_A[j-1] - unique_A[i])
        # Implicit else: if current_count < M, it means `j` reached `U` but we still couldn't gather M elements.
        # In subsequent iterations, `i` will increase, `current_count` will not increase (it might decrease), 
        # and `j` is already at `U`. So, the condition `current_count >= M` will never be met again.
        # The loop will continue until `i` reaches `U`, correctly handling this situation without needing an explicit `break`.

    # Print the overall minimum difference found.
    # Since N >= M is guaranteed by constraints, there exists at least one range covering M elements
    # (e.g., the range covering all elements includes N elements, and N >= M).
    # Thus, `min_diff` will always be updated from infinity to a finite value.
    print(min_diff)

# Execute the solve function to process standard input and produce standard output
solve()