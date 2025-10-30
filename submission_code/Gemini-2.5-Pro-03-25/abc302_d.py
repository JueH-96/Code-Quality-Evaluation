import sys
import bisect

def solve():
    # Read N, M, D from the first line of input.
    # N: number of gift candidates for Aoki
    # M: number of gift candidates for Snuke
    # D: maximum allowed absolute difference in values
    N, M, D = map(int, sys.stdin.readline().split())
    
    # Read the list of gift values for Aoki from the second line.
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read the list of gift values for Snuke from the third line.
    B = list(map(int, sys.stdin.readline().split()))

    # Sort both lists in ascending order. This is crucial for efficient searching.
    A.sort()
    B.sort()

    # Initialize the maximum sum found so far to -1. 
    # This value will be printed if no pair satisfying the condition is found.
    max_sum = -1

    # Optimization: Iterate through the potentially smaller list and perform binary search on the larger list.
    # This doesn't change the asymptotic time complexity but might offer a minor performance improvement.
    list1, list2 = A, B  # list1 will be the list we iterate through, list2 the list we search in.
    len1, len2 = N, M   # Keep track of the original lengths N and M associated with A and B.
    
    # If list A is actually larger than list B, swap them so that list1 is the shorter (or equal length) list.
    if N > M:
       list1, list2 = B, A # Swap the lists
       len1, len2 = M, N   # Update the lengths accordingly to match list1 and list2

    # Iterate through each element `val1` in the chosen list `list1`.
    for i in range(len1):
        val1 = list1[i]
        
        # We need to find an element `val2` in `list2` such that the absolute difference |val1 - val2| is at most D.
        # This condition is equivalent to: val1 - D <= val2 <= val1 + D.
        # Among all such `val2`, we want the one that maximizes the sum `val1 + val2`.
        # Since `val1` is fixed in this iteration, this is equivalent to maximizing `val2`.
        
        # We need to find the largest `val2` in `list2` such that `val2` is in the range `[val1 - D, val1 + D]`.
        
        # Use binary search (`bisect_right`) to find the insertion point `k` for `val1 + D` in `list2`.
        # `bisect_right(list2, x)` returns an index `k` such that all `list2[j]` for `j < k` have `list2[j] <= x`,
        # and all `list2[j]` for `j >= k` have `list2[j] > x`.
        k = bisect.bisect_right(list2, val1 + D)
        
        # If k is 0, it means all elements in `list2` are strictly greater than `val1 + D`.
        # In this case, no element `val2` can satisfy `val2 <= val1 + D`, so no element can be in the required range.
        # We continue to the next `val1`.
        if k == 0:
            continue

        # If k > 0, the element at index `k-1`, i.e., `list2[k-1]`, is the largest element in `list2`
        # that satisfies `list2[k-1] <= val1 + D`.
        # This element is our best candidate for `val2` to maximize the sum. Let's call it `potential_val2`.
        potential_val2 = list2[k-1]

        # Now we must check if this candidate `potential_val2` also satisfies the lower bound requirement:
        # `potential_val2 >= val1 - D`.
        if potential_val2 >= val1 - D:
            # If `potential_val2` satisfies both `potential_val2 <= val1 + D` (by selection) 
            # and `potential_val2 >= val1 - D` (by check), then it is within the valid range `[val1 - D, val1 + D]`.
            # Since it's the largest element satisfying the upper bound, it is the largest possible `val2`
            # in `list2` for the current `val1` that meets the difference constraint.
            # We have found a valid pair (`val1`, `potential_val2`).
            
            # Calculate the sum of the pair's values.
            current_sum = val1 + potential_val2
            
            # Update `max_sum` if the current sum is greater than the maximum sum found so far.
            if current_sum > max_sum:
                max_sum = current_sum

    # After iterating through all elements of `list1`, `max_sum` holds the maximum sum found among all valid pairs.
    # If no valid pair was ever found, `max_sum` will remain at its initial value of -1.
    # Print the final result.
    print(max_sum)

# Execute the main function to solve the problem.
solve()