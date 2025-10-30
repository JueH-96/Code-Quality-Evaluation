def max_experience_points(N, A):
    # Initialize DP table
    # dp[i][j] where i is the monster index, j is the count of defeated monsters modulo 2
    # j can be 0 or 1
    # Initialize with -infinity
    dp = [[-1] * 2 for _ in range(N+1)]
    dp[0][0] = 0  # Start with 0 defeated monsters, count modulo 2 is 0
    
    for i in range(1, N+1):
        a = A[i-1]
        for prev_count_mod in range(2):
            if dp[i-1][prev_count_mod] == -1:
                continue
            # Option 1: Let the monster go
            if dp[i][prev_count_mod] < dp[i-1][prev_count_mod]:
                dp[i][prev_count_mod] = dp[i-1][prev_count_mod]
            # Option 2: Defeat the monster
            new_count_mod = (prev_count_mod + 1) % 2
            additional = a if new_count_mod == 0 else 0
            total = dp[i-1][prev_count_mod] + a + additional
            if dp[i][new_count_mod] < total:
                dp[i][new_count_mod] = total
    # The maximum is the maximum of the last row
    return max(dp[N][0], dp[N][1])

# Read input
N = int(input())
A = list(map(int, input().split()))
# Compute and print the result
print(max_experience_points(N, A))