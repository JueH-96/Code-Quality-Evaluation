# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Sort the array to easily identify min and max values
A_sorted = sorted(A)

# Try all possibilities of removing i smallest and (K-i) largest elements
min_range = float('inf')
for i in range(K + 1):
    # After removing i smallest and (K-i) largest elements:
    # - The new minimum value is at index i
    # - The new maximum value is at index (N - 1) - (K - i)
    left_idx = i
    right_idx = N - 1 - (K - i)
    
    # Make sure we have at least one element left
    if left_idx <= right_idx:
        current_range = A_sorted[right_idx] - A_sorted[left_idx]
        min_range = min(min_range, current_range)

# Print the result
print(min_range)