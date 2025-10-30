def min_fatigue(N, presses):
    # Initialize the DP table
    dp = [[float('inf')] * 101 for _ in range(N + 1)]
    dp[0][0] = 0  # Starting with both hands at key 0 (not on any key)

    for i in range(1, N + 1):
        A_i, S_i = presses[i - 1]
        for left in range(101):
            for right in range(101):
                if dp[i - 1][left] == float('inf') and dp[i - 1][right] == float('inf'):
                    continue
                
                # If pressing with left hand
                if S_i == 'L':
                    dp[i][A_i] = min(dp[i][A_i], dp[i - 1][left] + abs(left - A_i))
                
                # If pressing with right hand
                if S_i == 'R':
                    dp[i][right] = min(dp[i][right], dp[i - 1][right] + abs(right - A_i))

                # If we don't move the hand, just keep the previous state
                dp[i][left] = min(dp[i][left], dp[i - 1][left])
                dp[i][right] = min(dp[i][right], dp[i - 1][right])

    # Find the minimum fatigue level after all presses
    min_fatigue_level = float('inf')
    for left in range(101):
        for right in range(101):
            min_fatigue_level = min(min_fatigue_level, dp[N][left] + dp[N][right])

    return min_fatigue_level

import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
presses = [tuple(map(str, line.split())) for line in data[1:N + 1]]
presses = [(int(A), S) for A, S in presses]

result = min_fatigue(N, presses)
print(result)