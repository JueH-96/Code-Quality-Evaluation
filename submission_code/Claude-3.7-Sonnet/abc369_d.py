def max_experience(N, A):
    # dp[j] represents the maximum experience points after considering
    # the first i monsters and defeating exactly j of them
    dp = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(min(i, N), 0, -1):
            # If defeating the current monster as the j-th defeated monster
            if j % 2 == 0:  # Even-numbered defeated monster (2nd, 4th, ...)
                dp[j] = max(dp[j], dp[j-1] + 2 * A[i-1])
            else:  # Odd-numbered defeated monster (1st, 3rd, ...)
                dp[j] = max(dp[j], dp[j-1] + A[i-1])
    
    return max(dp)

N = int(input())
A = list(map(int, input().split()))
print(max_experience(N, A))