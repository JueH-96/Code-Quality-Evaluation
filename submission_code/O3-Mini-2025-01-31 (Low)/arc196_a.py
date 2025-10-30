def main():
    import sys
    input = sys.stdin.readline
    n = int(input().strip())
    A = list(map(int, input().split()))
    
    # dp[i] will be max total score achievable using the prefix A[0..i-1] 
    # with dp[0] = 0 for an empty prefix, dp[1] = 0 (since no adjacent pair exists)
    dp0 = 0  # dp[i-2]
    dp1 = 0  # dp[i-1]
    
    for i in range(2, n+1):
        # Option 1: not pairing the last two, keep dp1.
        # Option 2: pair the last two numbers: dp[i-2] + |A[i-1] - A[i-2]|
        pair_score = dp0 + abs(A[i-1] - A[i-2])
        dp_current = pair_score if pair_score > dp1 else dp1
        
        dp0, dp1 = dp1, dp_current
    
    sys.stdout.write(str(dp1))

if __name__ == '__main__':
    main()