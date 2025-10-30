from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        K = len(set(nums))
        if K == 0:
            return 0
        
        def atMost(target):
            count = {}
            res = 0
            left = 0
            for right in range(len(nums)):
                x = nums[right]
                count[x] = count.get(x, 0) + 1
                while len(count) > target:
                    y = nums[left]
                    count[y] -= 1
                    if count[y] == 0:
                        del count[y]
                    left += 1
                res += right - left + 1
            return res
        
        return atMost(K) - atMost(K - 1)