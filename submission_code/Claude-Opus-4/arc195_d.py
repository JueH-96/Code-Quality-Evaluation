def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    # Dynamic programming approach
    # dp[i] = minimum operations to clear first i elements
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(n):
        # Option 1: Start a new group from position i
        # Find the rightmost position where we can extend with the same value
        val = a[i]
        j = i
        swaps = 0
        
        while j < n:
            if a[j] == val:
                # Cost to bring a[j] to position after previous same values
                # is the number of different values between
                dp[j + 1] = min(dp[j + 1], dp[i] + swaps + 1)
                j += 1
            else:
                swaps += 1
                j += 1
    
    return dp[n]

t = int(input())
for _ in range(t):
    print(solve())