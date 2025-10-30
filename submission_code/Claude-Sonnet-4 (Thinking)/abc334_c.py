n, k = map(int, input().split())
lost = list(map(int, input().split()))

# Count remaining socks for each color
sock_count = [2] * (n + 1)
sock_count[0] = 0

lost_set = set(lost)
for color in lost_set:
    sock_count[color] -= 1

# Collect unpaired socks
unpaired_socks = []

for color in range(1, n + 1):
    count = sock_count[color]
    remaining = count % 2
    if remaining > 0:
        unpaired_socks.append(color)

# Sort unpaired socks
unpaired_socks.sort()

# Calculate number of pairs we need to make from unpaired socks
total_socks_remaining = 2 * n - k
total_pairs_to_make = total_socks_remaining // 2
pairs_from_same_color = sum(sock_count[color] // 2 for color in range(1, n + 1))
additional_pairs_needed = total_pairs_to_make - pairs_from_same_color

# DP to find minimum weirdness for pairing unpaired socks
m = len(unpaired_socks)
if additional_pairs_needed == 0:
    print(0)
else:
    dp = [[float('inf')] * (additional_pairs_needed + 1) for _ in range(m + 1)]
    
    # Base case
    for i in range(m + 1):
        dp[i][0] = 0
    
    for i in range(1, m + 1):
        for j in range(1, min(i // 2 + 1, additional_pairs_needed + 1)):
            # Don't include sock i-1
            dp[i][j] = dp[i - 1][j]
            
            # Pair sock i-1 with sock i-2
            if i >= 2:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 1] + abs(unpaired_socks[i - 1] - unpaired_socks[i - 2]))
    
    print(dp[m][additional_pairs_needed])