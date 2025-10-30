import sys

def solve():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))

    # Initialize max_overall and current_max_ending_here with the first element.
    # This correctly handles the n=1 case and ensures a non-empty subarray is considered.
    # Since n >= 1 is guaranteed, a[0] always exists.
    max_overall = a[0]
    current_max_ending_here = a[0]

    # Iterate from the second element to the end of the array.
    for i in range(1, n):
        # Determine the parities of the current and previous elements.
        # Python's '%' operator works correctly for both positive and negative integers
        # to determine parity (0 for even, 1 for odd).
        prev_parity = a[i-1] % 2
        curr_parity = a[i] % 2

        if prev_parity != curr_parity:
            # If parities are different, a[i] can extend the current subarray.
            # Apply Kadane's algorithm logic:
            # Either extend the previous max sum (current_max_ending_here + a[i])
            # or start a new subarray with a[i] (if extending makes the sum smaller).
            current_max_ending_here = max(a[i], current_max_ending_here + a[i])
        else:
            # If parities are the same, a[i] cannot extend the previous subarray.
            # It must start a new subarray by itself.
            current_max_ending_here = a[i]
        
        # Update the overall maximum sum found so far.
        max_overall = max(max_overall, current_max_ending_here)
    
    # Output the result for the current test case.
    sys.stdout.write(str(max_overall) + "
")

# Read the number of test cases.
t = int(sys.stdin.readline())

# Run the solve function for each test case.
for _ in range(t):
    solve()