class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        n = len(nums)
        
        for i in range(n - 1):
            if nums[i + 1] - nums[i] == 1:
                current_length = 2
                for j in range(i + 2, n):
                    if nums[j] == nums[j - 2]:
                        current_length += 1
                    else:
                        break
                max_length = max(max_length, current_length)
        
        return max_length