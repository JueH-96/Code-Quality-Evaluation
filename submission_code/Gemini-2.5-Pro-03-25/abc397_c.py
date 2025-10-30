# YOUR CODE HERE
import sys

def solve():
    # Read N, the length of the sequence
    n = int(sys.stdin.readline())
    # Read the sequence A as a list of integers
    a = list(map(int, sys.stdin.readline().split()))

    # Calculate prefix distinct counts
    # prefix_distinct[k] will store the number of distinct elements in the subarray A[0...k]
    prefix_distinct = [0] * n
    # Use a set to keep track of elements seen so far in the prefix
    seen_prefix = set()
    for i in range(n):
        seen_prefix.add(a[i])
        prefix_distinct[i] = len(seen_prefix)

    # Calculate suffix distinct counts
    # suffix_distinct[k] will store the number of distinct elements in the subarray A[k...N-1]
    suffix_distinct = [0] * n
    # Use a set to keep track of elements seen so far in the suffix (iterating backwards)
    seen_suffix = set()
    for i in range(n - 1, -1, -1):
        seen_suffix.add(a[i])
        suffix_distinct[i] = len(seen_suffix)

    # Initialize the maximum sum found so far
    max_sum = 0

    # Iterate through all possible split points.
    # The split occurs after index i-1, meaning the left subarray is A[0...i-1]
    # and the right subarray is A[i...N-1].
    # The loop variable 'i' represents the starting index of the right subarray.
    # The loop runs for i from 1 to n-1. This ensures that both subarrays are non-empty.
    # Left subarray length is i, right subarray length is n-i. Minimum length is 1.
    for i in range(1, n):
        # Count of distinct elements in the left subarray A[0...i-1] is prefix_distinct[i-1].
        # Count of distinct elements in the right subarray A[i...n-1] is suffix_distinct[i].
        current_sum = prefix_distinct[i-1] + suffix_distinct[i]
        
        # Update the maximum sum if the current sum is greater
        if current_sum > max_sum:
            max_sum = current_sum
            
    # Alternatively, using the max() function:
    # max_sum = 0
    # for i in range(1, n):
    #     max_sum = max(max_sum, prefix_distinct[i-1] + suffix_distinct[i])

    # Print the final maximum sum to standard output
    print(max_sum)

# Call the solve function to execute the main logic
solve()