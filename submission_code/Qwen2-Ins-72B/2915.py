from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        prefix_sum = defaultdict(int)
        prefix_sum[0] = 1
        cnt, res = 0, 0
        for num in nums:
            if num % modulo == k:
                cnt += 1
            res += prefix_sum[(cnt - k) % modulo]
            prefix_sum[cnt % modulo] += 1
        return res