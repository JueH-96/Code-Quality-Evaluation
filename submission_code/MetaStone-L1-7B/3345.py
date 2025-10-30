MOD = 10**9 + 7

def sumOfPower(nums, k):
    if not nums:
        return 0
    inv_2 = pow(2, MOD-2, MOD)
    dp = [0] * (k + 1)
    dp[0] = 1  # Base case: empty subset contributes 1/(2^0) = 1
    for x in nums:
        for s in range(k, x - 1, -1):
            if s - x >= 0:
                dp[s] = (dp[s] + inv_2 * dp[s - x]) % MOD
    total = (pow(2, len(nums), MOD) * dp[k]) % MOD
    return total