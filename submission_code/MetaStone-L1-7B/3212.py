MOD = 10**9 + 7

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        last_occurrence = {}
        dp = [0] * (n + 1)
        prefix = [0] * (n + 1)
        dp[0] = 1
        prefix[0] = 1
        s = 0  # earliest start

        for i in range(1, n + 1):
            current_num = nums[i-1]
            if current_num in last_occurrence:
                if last_occurrence[current_num] >= s:
                    s = last_occurrence[current_num] + 1
            last_occurrence[current_num] = i - 1  # 0-based index for nums

            if s == 0:
                dp[i] = prefix[s]  # prefix[s] is prefix[0]
            else:
                dp[i] = prefix[s-1]

            # Update the prefix sum array
            prefix[i] = (prefix[i-1] + dp[i]) % MOD

        return dp[n] % MOD