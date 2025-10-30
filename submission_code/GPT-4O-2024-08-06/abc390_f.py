def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Initialize dp array
    dp = [[0] * N for _ in range(N)]
    
    # Base case: single element subarrays
    for i in range(N):
        dp[i][i] = 1
    
    # Fill dp table
    for length in range(2, N + 1):  # length of the subarray
        for L in range(N - length + 1):
            R = L + length - 1
            # Start with the worst case: erase one by one
            dp[L][R] = dp[L][R - 1] + 1
            # Try to find a better partition
            seen = set()
            for k in range(R, L - 1, -1):
                if A[k] in seen:
                    continue
                seen.add(A[k])
                if all(x in seen for x in range(min(seen), max(seen) + 1)):
                    if k == R:
                        dp[L][R] = min(dp[L][R], dp[L][k - 1] + 1)
                    else:
                        dp[L][R] = min(dp[L][R], dp[L][k - 1] + dp[k][R])
    
    # Calculate the result
    result = 0
    for L in range(N):
        for R in range(L, N):
            result += dp[L][R]
    
    print(result)