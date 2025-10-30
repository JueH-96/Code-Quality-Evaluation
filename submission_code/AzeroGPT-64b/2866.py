class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        start = -1
        res = 0
        n = len(nums)
        is_even = False
        
        for i in range(n):
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                if not is_even:
                    is_even = True
                    start = i
                res = max(res, i - start + 1)
            elif nums[i] % 2 != 0 and i > 0 and nums[i-1] % 2 == 0 and nums[i] <= threshold:
                if not is_even:
                    is_even = True
                    start = i-1
                res = max(res, i - start + 1)
            else:
                is_even = False
        
        return res