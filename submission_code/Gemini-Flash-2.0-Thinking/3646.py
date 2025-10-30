class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        dp_sum = {}
        dp_count = {}

        for num in nums:
            # Contribution from the subsequence [num]
            current_sum = dp_sum.get(num, 0)
            dp_sum[num] = (current_sum + num) % MOD
            dp_count[num] = dp_count.get(num, 0) + 1

            # Extend good subsequences ending with num - 1
            if num - 1 in dp_sum:
                prev_sum = dp_sum[num - 1]
                prev_count = dp_count[num - 1]

                current_sum = dp_sum.get(num, 0)
                dp_sum[num] = (current_sum + prev_sum + prev_count * num) % MOD

                current_count = dp_count.get(num, 0)
                dp_count[num] = (current_count + prev_count) % MOD

        total_sum = 0
        for sum_val in dp_sum.values():
            total_sum = (total_sum + sum_val) % MOD

        return total_sum