MOD = 998244353

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    points = []
    
    index = 1
    for i in range(N):
        x = int(data[index])
        y = int(data[index + 1])
        points.append((x, y))
        index += 2
    
    # Sort points by x coordinate (since x is a permutation of 1 to N, this is effectively sorting by index)
    points.sort()
    
    # dp[i] will store the number of valid sets that end at point i
    dp = [1] * N
    
    # Calculate dp values
    for i in range(N):
        for j in range(i):
            if points[j][1] < points[i][1]:  # point j can be dominated by point i
                dp[i] = (dp[i] + dp[j]) % MOD
    
    # The result is the sum of all dp values
    result = sum(dp) % MOD
    print(result)