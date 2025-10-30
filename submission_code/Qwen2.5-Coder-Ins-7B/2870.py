class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        n = len(nums)
        
        for i in range(n - 1):
            if nums[i + 1] - nums[i] == 1:
                length = 2
                j = i + 1
                while j + 1 < n and ((j - i) % 2 == 0 and nums[j + 1] == nums[j] - 1) or ((j - i) % 2 == 1 and nums[j + 1] == nums[j] + 1):
                    length += 1
                    j += 1
                max_length = max(max_length, length)
        
        return max_length