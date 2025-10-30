# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Read N and K from the first line of standard input
        n, k = map(int, sys.stdin.readline().split())
        
        # Read the sequence A from the second line
        a = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # This handles potential empty lines or formatting errors.
        return

    # Sort the array A. This is the most crucial step, as it allows us
    # to find the smallest range of values containing a certain number of elements.
    a.sort()
    
    # We need to keep N-K elements in the final sequence B.
    num_to_keep = n - k
    
    # The problem reduces to finding the contiguous subarray of length `num_to_keep`
    # in the sorted array `a` with the minimum difference between its last and first elements.

    # Initialize the minimum difference with the first possible window.
    # This window spans from index 0 to `num_to_keep - 1`.
    min_diff = a[num_to_keep - 1] - a[0]
    
    # Slide the window of size `num_to_keep` across the sorted array.
    # The number of windows is `n - num_to_keep + 1`, which equals `k + 1`.
    # We have already processed the first window (i=0), so we loop from i=1 to k.
    for i in range(1, k + 1):
        # The current window starts at index `i` and ends at `i + num_to_keep - 1`.
        diff = a[i + num_to_keep - 1] - a[i]
        
        # If this window gives a smaller difference, update our minimum.
        if diff < min_diff:
            min_diff = diff
            
    # Print the final minimum difference to standard output.
    print(min_diff)

if __name__ == "__main__":
    solve()