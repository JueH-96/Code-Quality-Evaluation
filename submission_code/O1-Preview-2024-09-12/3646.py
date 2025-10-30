class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        from collections import defaultdict
        MOD = 10**9 + 7
        count = defaultdict(int)
        sum_ = defaultdict(int)
        total_sum = 0
        for num in nums:
            cnt_new = 1  # Start a new subsequence with current num
            sum_new = num  # Sum of the new subsequence
            for adj in [num - 1, num + 1]:
                if adj in count:
                    cnt_new = (cnt_new + count[adj]) % MOD
                    sum_new = (sum_new + sum_[adj] + num * count[adj]) % MOD
            count[num] = (count[num] + cnt_new) % MOD
            sum_[num] = (sum_[num] + sum_new) % MOD
            total_sum = (total_sum + sum_new) % MOD
        return total_sum