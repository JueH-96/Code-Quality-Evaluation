from collections import defaultdict
from typing import List

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        count = defaultdict(int)
        sum_total = defaultdict(int)
        ans = 0
        for num in nums:
            left_count = count[num - 1]
            left_sum = sum_total[num - 1]
            right_count = count[num + 1]
            right_sum = sum_total[num + 1]
            new_count = left_count + right_count + 1
            new_sum = (left_sum + right_sum) + num * new_count
            ans = (ans + new_sum) % MOD
            count[num] += new_count
            sum_total[num] += new_sum
        return ans % MOD