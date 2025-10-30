class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        from collections import defaultdict

        MOD = 10 ** 9 + 7
        n = len(nums)
        max_sum = sum(nums)
        max_size = n

        dp = defaultdict(lambda: defaultdict(int))
        dp[0][0] = 1  # sum 0 with size 0 occurs once

        for num in nums:
            dp_new = defaultdict(lambda: defaultdict(int))
            for sum_s in dp:
                for size_c in dp[sum_s]:
                    cnt = dp[sum_s][size_c]
                    # Without current num
                    dp_new[sum_s][size_c] = (dp_new[sum_s][size_c] + cnt) % MOD
                    # With current num
                    sum_new = sum_s + num
                    size_new = size_c + 1
                    dp_new[sum_new][size_new] = (dp_new[sum_new][size_new] + cnt) % MOD
            dp = dp_new

        # Precompute powers of 2
        pow2 = [1] * (n + 1)
        for i in range(1, n +1):
            pow2[i] = (pow2[i -1 ] * 2) % MOD

        total_sum = 0
        for sum_s in dp:
            if sum_s == k:
                for size_c in dp[sum_s]:
                    cnt_c = dp[sum_s][size_c]
                    total_sum = (total_sum + cnt_c * pow2[n - size_c]) % MOD
        return total_sum