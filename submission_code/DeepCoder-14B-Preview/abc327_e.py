n = int(input())
p = list(map(int, input().split()))

dp = {0: (0.0, 0.0)}

for P in p:
    new_dp = {}
    # Copy current dp to new_dp
    for key in dp:
        new_dp[key] = dp[key]
    # Iterate through all m in current dp
    for m in list(dp.keys()):
        # Compute new_m
        if m == 0:
            new_m = 1
            new_S_num = 0.9 * dp[m][0] + P
            new_S_den = 0.9 * dp[m][1] + 1
        else:
            new_m = m + 1
            new_S_num = 0.9 * dp[m][0] + P
            new_S_den = 0.9 * dp[m][1] + 1
        # Update new_dp for new_m
        if new_m in new_dp:
            current_ratio = new_dp[new_m][0] / new_dp[new_m][1]
            new_ratio = new_S_num / new_S_den
            if new_ratio > current_ratio:
                new_dp[new_m] = (new_S_num, new_S_den)
        else:
            new_dp[new_m] = (new_S_num, new_S_den)
    # Update dp to new_dp
    dp = new_dp

max_R = -float('inf')

for m in dp:
    if m == 0:
        continue
    S_num, S_den = dp[m]
    if S_den == 0:
        continue
    ratio = S_num / S_den
    R = ratio - 1200 / (m ** 0.5)
    if R > max_R:
        max_R = R

print("{0:.15f}".format(max_R))