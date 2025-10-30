# YOUR CODE HERE
import sys

def solve():
    # Read N (number of gifts) and M (length of interval) from standard input
    N, M = map(int, sys.stdin.readline().split())
    
    # Read the coordinates of the N gifts from standard input
    A = list(map(int, sys.stdin.readline().split()))
    
    # Sort the gift coordinates in non-decreasing order.
    # Sorting is necessary for the efficient two-pointer approach.
    A.sort()
    
    # Initialize the maximum number of gifts found so far to 0.
    max_gifts = 0
    # Initialize the 'right' pointer for the two-pointer technique.
    # 'right' will represent the index of the first element *outside* the current interval.
    right = 0
    
    # Iterate through the sorted array using 'left' as the index of the potential start of the interval.
    # The interval considered starts at coordinate A[left].
    for left in range(N):
        # Calculate the target value, which is the exclusive upper bound of the interval.
        # The interval is [A[left], A[left] + M).
        target_val = A[left] + M
        
        # Advance the 'right' pointer as long as it's within the array bounds (right < N)
        # and the element at A[right] is less than the target value (A[right] < A[left] + M).
        # This loop finds the first index 'right' such that A[right] >= target_val.
        while right < N and A[right] < target_val:
            right += 1
        
        # The number of gifts included in the interval starting at A[left] is the
        # count of elements from index 'left' up to (but not including) 'right'.
        # This count is given by `right - left`.
        current_gifts = right - left
        
        # Update the overall maximum number of gifts found so far.
        max_gifts = max(max_gifts, current_gifts)
        
        # Optimization: If the 'right' pointer reaches the end of the array (index N),
        # it means that for the current A[left], all remaining gifts (from index 'left' to N-1)
        # are included in the interval. Any subsequent interval starting at a larger 'left'
        # index will necessarily contain fewer gifts (since 'right' cannot go beyond N).
        # Therefore, we can break the loop early as we've found the maximum possible count
        # involving the later elements.
        if right == N:
             break

    # Print the maximum number of gifts that can be acquired.
    print(max_gifts)

# Call the main function to solve the problem
solve()