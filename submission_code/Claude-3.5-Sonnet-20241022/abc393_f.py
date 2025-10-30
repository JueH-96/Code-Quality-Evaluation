def find_longest_increasing_subsequence(A, R, X):
    # Find the longest increasing subsequence up to index R
    # where all elements are at most X
    dp = []  # stores the smallest possible value for each length
    
    for i in range(R):
        if A[i] > X:
            continue
            
        # Binary search to find the position to insert A[i]
        left, right = 0, len(dp)
        while left < right:
            mid = (left + right) // 2
            if dp[mid] >= A[i]:
                right = mid
            else:
                left = mid + 1
                
        if left == len(dp):
            dp.append(A[i])
        else:
            dp[left] = A[i]
            
    return len(dp)

# Read input
N, Q = map(int, input().split())
A = list(map(int, input().split()))

# Process queries
for _ in range(Q):
    R, X = map(int, input().split())
    result = find_longest_increasing_subsequence(A, R, X)
    print(result)