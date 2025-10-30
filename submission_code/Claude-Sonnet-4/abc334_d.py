# YOUR CODE HERE
N, Q = map(int, input().split())
R = list(map(int, input().split()))

# Sort the reindeer requirements in ascending order
R.sort()

# Precompute prefix sums for efficiency
prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i + 1] = prefix_sum[i] + R[i]

# Process each query
for _ in range(Q):
    X = int(input())
    
    # Binary search to find the maximum number of sleighs
    left, right = 0, N
    answer = 0
    
    while left <= right:
        mid = (left + right) // 2
        if prefix_sum[mid] <= X:
            answer = mid
            left = mid + 1
        else:
            right = mid - 1
    
    print(answer)