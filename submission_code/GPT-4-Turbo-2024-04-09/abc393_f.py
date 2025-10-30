import sys
import bisect

input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
A = list(map(int, data[2:N+2]))

queries = []
index = N + 2
for _ in range(Q):
    R = int(data[index])
    X = int(data[index + 1])
    queries.append((R, X))
    index += 2

# To handle the queries efficiently, we need to preprocess the data
# We will use a list of lists to store the longest increasing subsequences up to each index
# This will be done using a dynamic programming approach with binary search for efficiency

# dp[i] will hold the smallest possible tail value of an increasing subsequence of length i+1
dp = [[] for _ in range(N + 1)]

for i in range(1, N + 1):
    dp[i] = dp[i - 1][:]  # Copy the previous state
    num = A[i - 1]
    # Find the position to replace or extend in dp array
    pos = bisect.bisect_right(dp[i], num)
    if pos < len(dp[i]):
        dp[i][pos] = num
    else:
        dp[i].append(num)

# Now, we need to answer each query
results = []
for R, X in queries:
    # We need to find the longest increasing subsequence in dp[R] that all elements are <= X
    # We can do this using binary search to find the largest length where the tail is <= X
    l = 0
    r = len(dp[R])
    while l < r:
        mid = (l + r) // 2
        if dp[R][mid] <= X:
            l = mid + 1
        else:
            r = mid
    results.append(l)

# Print the results for each query
for result in results:
    print(result)