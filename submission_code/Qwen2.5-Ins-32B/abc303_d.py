import sys

def min_time_to_string(x, y, z, s):
    n = len(s)
    dp = [[float('inf')] * 2 for _ in range(n + 1)]
    dp[0][0] = 0  # Initial state with Caps Lock off
    dp[0][1] = z  # Initial state with Caps Lock on

    for i in range(n):
        char = s[i]
        for caps_lock in range(2):
            if char == 'a':
                if caps_lock == 0:
                    dp[i + 1][0] = min(dp[i + 1][0], dp[i][0] + x)
                    dp[i + 1][1] = min(dp[i + 1][1], dp[i][1] + y)
                else:
                    dp[i + 1][0] = min(dp[i + 1][0], dp[i][0] + y)
                    dp[i + 1][1] = min(dp[i + 1][1], dp[i][1] + x)
            else:  # char == 'A'
                if caps_lock == 0:
                    dp[i + 1][0] = min(dp[i + 1][0], dp[i][0] + y)
                    dp[i + 1][1] = min(dp[i + 1][1], dp[i][1] + x)
                else:
                    dp[i + 1][0] = min(dp[i + 1][0], dp[i][0] + x)
                    dp[i + 1][1] = min(dp[i + 1][1], dp[i][1] + y)
            dp[i + 1][0] = min(dp[i + 1][0], dp[i][1] + z)
            dp[i + 1][1] = min(dp[i + 1][1], dp[i][0] + z)

    return min(dp[n][0], dp[n][1])

input = sys.stdin.read
data = input().split()
x, y, z = map(int, data[0].split())
s = data[1]
print(min_time_to_string(x, y, z, s))