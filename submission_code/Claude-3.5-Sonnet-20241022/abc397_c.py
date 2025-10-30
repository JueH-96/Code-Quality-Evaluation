# Read input
N = int(input())
A = list(map(int, input().split()))

# Function to find maximum sum of distinct integers in two subarrays
def max_distinct_sum():
    max_sum = 0
    
    # Try all possible split positions
    for i in range(1, N):
        # Count distinct elements in left subarray (1 to i)
        left_distinct = len(set(A[:i]))
        
        # Count distinct elements in right subarray (i+1 to N)
        right_distinct = len(set(A[i:]))
        
        # Update maximum sum if current split gives better result
        max_sum = max(max_sum, left_distinct + right_distinct)
    
    return max_sum

# Print result
print(max_distinct_sum())