import sys

# Read N, the number of bean types
N = int(sys.stdin.readline())

# Dictionary to store the minimum deliciousness for each color
# Key: color_id (int), Value: min_deliciousness (int)
min_deliciousness_for_color = {}

# A_i is at most 10^9. This value is used as a stand-in for infinity
# when a color is encountered for the first time.
# Any value strictly greater than 10^9 would work, e.g., 10**9 + 1.
# float('inf') could also be used.
INITIAL_MAX_VALUE_FOR_MIN_COMPARISON = 10**9 + 7 

# Process each bean type
for _ in range(N):
    # Read deliciousness A and color C for the current bean type
    A, C = map(int, sys.stdin.readline().split())
    
    # Get the current minimum deliciousness recorded for color C.
    # If color C has not been seen before, .get() returns INITIAL_MAX_VALUE_FOR_MIN_COMPARISON.
    current_min_for_this_color = min_deliciousness_for_color.get(C, INITIAL_MAX_VALUE_FOR_MIN_COMPARISON)
    
    # Update the minimum deliciousness for color C if the current bean's A is smaller.
    min_deliciousness_for_color[C] = min(current_min_for_this_color, A)

# After processing all beans, min_deliciousness_for_color contains the
# minimum deliciousness for each available color (M_C for each C).
# We need to find the maximum among these minimums.
#
# Constraints state N >= 1, so min_deliciousness_for_color will not be empty.
# Also, A_i >= 1, so all deliciousness values stored will be >= 1.
ans = max(min_deliciousness_for_color.values())

# Print the result
print(ans)