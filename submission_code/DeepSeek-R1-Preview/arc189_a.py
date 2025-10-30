mod = 998244353

n = int(input())
a = list(map(int, input().split()))

initial = [(i + 1) % 2 for i in range(n)]

runs = []
if not a:
    print(1)
    exit()

current = a[0]
start = 0
for i in range(1, n):
    if a[i] != current:
        runs.append((start, i-1, current))
        current = a[i]
        start = i
runs.append((start, n-1, current))

# Check feasibility
for s, e, v in runs:
    if initial[s] != v or initial[e] != v:
        print(0)
        exit()

# Now, for each run, compute the number of ways
# dp[i] represents the number of ways for the first i elements
dp = [1] * (n + 1)
for i in range(1, n):
    dp[i + 1] = dp[i]
    if a[i] == a[i - 1]:
        dp[i + 1] = (dp[i + 1] + dp[i - 1]) % mod

# For each run, compute the product
result = 1
for s, e, v in runs:
    length = e - s + 1
    if length == 1:
        continue
    # The number of ways for this run is dp[length - 1]
    # Wait, need to adjust indices
    # Let's consider the run from 0 to length-1
    ways = 1
    current_run_dp = [0] * (length + 1)
    current_run_dp[0] = 1
    for i in range(1, length):
        current_run_dp[i + 1] = current_run_dp[i]
        # Check if we can merge with the previous
        if i >= 1:
            current_run_dp[i + 1] = (current_run_dp[i + 1] + current_run_dp[i - 1]) % mod
    ways = current_run_dp[length]
    result = (result * ways) % mod

print(result)