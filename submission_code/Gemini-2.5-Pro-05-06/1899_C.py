import sys

# Solution logic for a single test case
def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize max_so_far with the value of the first element.
    # A single element forms a valid non-empty subarray.
    max_so_far = A[0]
    
    # current_sum_ending_here stores the maximum sum of an alternating-parity 
    # subarray that ends at the current element A[i] (or A[i-1] before update in loop).
    current_sum_ending_here = A[0]

    # Iterate through the array starting from the second element
    for i in range(1, N):
        # Get parities of the current element A[i] and the previous element A[i-1].
        # Python's % operator: x % 2 is 0 for even, 1 for odd (for both positive and negative x).
        parity_current = A[i] % 2
        parity_previous = A[i-1] % 2
        
        if parity_current != parity_previous:
            # Parities are different. We can potentially extend the subarray ending at A[i-1].
            # The maximum sum for a subarray ending at A[i] would be either:
            # 1. A[i] itself (if current_sum_ending_here from A[i-1] is negative or not worth adding).
            # 2. A[i] added to current_sum_ending_here (extending the previous valid subarray).
            current_sum_ending_here = max(A[i], current_sum_ending_here + A[i])
        else:
            # Parities are the same. The subarray ending at A[i-1] cannot be extended with A[i]
            # while maintaining the alternating parity condition.
            # Therefore, any valid alternating-parity subarray ending at A[i] must start with A[i].
            current_sum_ending_here = A[i]
        
        # Update the overall maximum sum found so far.
        if current_sum_ending_here > max_so_far:
            max_so_far = current_sum_ending_here
            
    # Print the result for the current test case
    sys.stdout.write(str(max_so_far) + "
")

# Read the number of test cases
num_test_cases = int(sys.stdin.readline())
# Process each test case
for _ in range(num_test_cases):
    solve()