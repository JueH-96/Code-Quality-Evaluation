# YOUR CODE HERE
import sys

# Function to solve a single test case
def solve():
    # Read the size of the array
    n = int(sys.stdin.readline())
    # Read the array elements as integers
    a = list(map(int, sys.stdin.readline().split()))

    # Initialize the maximum sum found so far (global_max) and the maximum sum
    # ending at the current position (current_max) with the value of the first element.
    # This initialization correctly handles the case n=1 and ensures that if all elements 
    # are negative, the largest (least negative) element is considered as a potential maximum sum.
    global_max = a[0] 
    current_max = a[0]

    # Iterate through the array starting from the second element (index 1) up to n-1.
    for k in range(1, n):
        # Check if the parity of the current element a[k] differs from the previous element a[k-1].
        # The modulo operator `% 2` in Python gives 0 for even numbers and 1 for odd numbers (including negative odds).
        # Example: 4 % 2 = 0, 5 % 2 = 1, -2 % 2 = 0, -3 % 2 = 1.
        # So `(a[k] % 2) != (a[k-1] % 2)` correctly checks if parities are different.
        if (a[k] % 2) != (a[k-1] % 2):
            # If parities differ, we can potentially extend the valid alternating parity subarray ending at k-1.
            # We apply Kadane's algorithm logic: The maximum sum ending at k is the maximum of:
            # 1. Starting a new subarray with just a[k].
            # 2. Extending the maximum sum valid subarray ending at k-1 by adding a[k].
            # `max(a[k], current_max + a[k])` implements this choice.
            current_max = max(a[k], current_max + a[k])
        else:
            # If parities are the same, the constraint is violated if we extend the previous subarray.
            # Therefore, any valid alternating parity subarray ending at k must start at k itself.
            # The maximum sum ending at k in this case is simply a[k].
            current_max = a[k]
        
        # After calculating the maximum sum of a valid subarray ending at k (`current_max`),
        # update the overall maximum sum found so far (`global_max`).
        global_max = max(global_max, current_max)

    # Print the overall maximum sum found across all possible valid subarrays.
    print(global_max)

# Read the total number of test cases
T = int(sys.stdin.readline())
# Iterate through each test case and call the solve function
for _ in range(T):
    solve()