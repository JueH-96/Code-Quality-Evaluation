def count_distinct(arr):
    return len(set(arr))

# Read input
N = int(input())
A = list(map(int, input().split()))

# Initialize answer
max_sum = 0

# Try all possible splits (i,j)
for i in range(1, N-1):  # First split position
    for j in range(i+1, N):  # Second split position
        # Split into three subarrays
        first = A[:i]  # A_1 to A_i
        second = A[i:j]  # A_{i+1} to A_j
        third = A[j:]  # A_{j+1} to A_N
        
        # Count distinct elements in each subarray
        distinct_sum = (count_distinct(first) + 
                       count_distinct(second) + 
                       count_distinct(third))
        
        # Update maximum if current sum is larger
        max_sum = max(max_sum, distinct_sum)

# Print answer
print(max_sum)