# YOUR CODE HERE
import bisect

def longest_increasing_subsequence(arr):
    if not arr:
        return 0
    
    # dp[i] will store the smallest tail element for LIS of length i+1
    dp = []
    
    for num in arr:
        # Find the position where num should be inserted
        pos = bisect.bisect_left(dp, num)
        
        if pos == len(dp):
            dp.append(num)
        else:
            dp[pos] = num
    
    return len(dp)

# Read input
N, Q = map(int, input().split())
A = list(map(int, input().split()))

# Process queries
for _ in range(Q):
    R, X = map(int, input().split())
    
    # Consider only first R elements and filter those <= X
    filtered = []
    for i in range(R):
        if A[i] <= X:
            filtered.append(A[i])
    
    # Find LIS of filtered array
    result = longest_increasing_subsequence(filtered)
    print(result)