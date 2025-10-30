import sys
input = sys.stdin.readline

def solve():
    n = int(input())
    a = list(map(int, input().split()))

    # current_max: maximum sum of a valid alternating-parity subarray ending at the current position
    # global_max: overall maximum sum found so far
    
    # Since the subarray must be non-empty and n >= 1, we can initialize with the first element.
    # A single element subarray is always valid.
    current_max = a[0]
    global_max = a[0]

    # Iterate through the array starting from the second element
    for i in range(1, n):
        # Check if parities of a[i] and a[i-1] are different
        # Parity check using modulo 2 on absolute value: 0 for even, 1 for odd
        # (abs(x) % 2) != (abs(y) % 2) is True if x and y have different parities
        # This handles positive, negative, and zero values correctly.
        
        # Check if a[i] and a[i-1] have different parities
        if (abs(a[i]) % 2) != (abs(a[i-1]) % 2):
            # If parities are different, we can potentially extend the previous valid subarray.
            # The new current_max is the maximum sum ending at index i. This can be either:
            # 1. Starting a new subarray at a[i]. The sum is a[i]. (This is always a valid alternating subarray of length 1)
            # 2. Extending the previous valid subarray (ending at i-1) by adding a[i]. The sum is current_max + a[i].
            # We take the maximum of these two options.
            current_max = max(a[i], current_max + a[i])
        else:
            # If parities are the same, we cannot extend the valid alternating-parity
            # subarray ending at i-1 with a[i].
            # Thus, any valid alternating-parity subarray ending at i must start at i.
            # The maximum sum ending at i is just the element a[i] itself.
            current_max = a[i]

        # Update the overall maximum sum found across all valid subarrays encountered so far
        global_max = max(global_max, current_max)

    # Print the result for this test case
    print(global_max)

# Read the number of test cases
t = int(input())

# Process each test case
for _ in range(t):
    solve()