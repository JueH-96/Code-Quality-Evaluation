mod = 998244353

n = int(input())
A = list(map(int, input().split()))

dp = [0] * 1024
dp[0] = 1  # Initial state: no sums, probability 1

for a in A:
    inv_a = pow(a, mod - 2, mod)
    new_dp = [0] * 1024
    for state in range(1024):
        if dp[state] == 0:
            continue
        # Handle x > 10
        count_x_gt_10 = max(0, a - 10)
        if count_x_gt_10 > 0:
            transition = dp[state] * count_x_gt_10 % mod
            transition = transition * inv_a % mod
            new_dp[state] = (new_dp[state] + transition) % mod
        # Handle x <= 10 and x <= a
        max_x = min(10, a)
        for x in range(1, max_x + 1):
            if x == 10:
                continue
            # Check if any sum s in current state where s + x == 10
            target = 10 - x
            invalid = False
            if 1 <= target <= 9:
                if (state & (1 << (target - 1))):
                    invalid = True
            if invalid:
                continue
            # Compute new_bitmask
            new_bitmask = state
            new_bitmask |= (1 << (x - 1))  # Add x
            # Add s + x for each s in state's sums (s is 1-9)
            for s in range(1, 10):
                if (state & (1 << (s - 1))):
                    t = s + x
                    if t < 10:
                        new_bitmask |= (1 << (t - 1))
            # Update new_dp
            transition = dp[state] * inv_a % mod
            new_dp[new_bitmask] = (new_dp[new_bitmask] + transition) % mod
    dp = new_dp

sum_dp = sum(dp) % mod
ans = (1 - sum_dp) % mod
print(ans)