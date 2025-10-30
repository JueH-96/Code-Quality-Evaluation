class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_length = 0
        i = 0
        
        while i < n:
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                start = i
                while i + 1 < n and nums[i] % 2 != nums[i+1] % 2 and nums[i+1] <= threshold:
                    i += 1
                max_length = max(max_length, i - start + 1)
            i += 1
        
        return max_length