import sys

# Read all input data
data = sys.stdin.read().split()
index = 0
T = data[index]
index += 1
N = int(data[index])
index += 1

# Read bags
bags = []
for _ in range(N):
    A = int(data[index])
    index += 1
    bag = []
    for _ in range(A):
        s = data[index]
        index += 1
        bag.append(s)
    bags.append(bag)

# Length of T
len_T = len(T)

# Define a large number for infinity
INF = 10**9

# Initialize DP table
dp = [[INF for _ in range(len_T + 1)] for _ in range(N + 1)]
dp[0][0] = 0

# Fill DP table
for i in range(1, N + 1):
    # Option to do nothing for all j
    for j in range(0, len_T + 1):
        dp[i][j] = dp[i - 1][j]
    
    # Option to add a string from bag i (bags[i-1])
    for str_val in bags[i - 1]:
        L = len(str_val)
        for end_j in range(L, len_T + 1):
            if T[end_j - L : end_j] == str_val:
                dp[i][end_j] = min(dp[i][end_j], dp[i - 1][end_j - L] + 1)

# Get the answer
ans = dp[N][len_T]
if ans >= INF:
    print(-1)
else:
    print(ans)