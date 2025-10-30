from collections import Counter
from itertools import accumulate

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        distinct = len(set(nums))
        prefix_nums = [0] + list(accumulate(nums, func=lambda x, y: x | (1 << y), initial=0))
        ans = 0

        for left in range(n):
            for right in range(left + distinct, n + 1):
                if bin(prefix_nums[right] - prefix_nums[left]).count('1') == distinct:
                    ans += 1
                    break
                
        return ans