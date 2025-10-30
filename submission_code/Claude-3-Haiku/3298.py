class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        
        max_count = 1
        for i in range(n-1):
            if nums[i+1] - nums[i] <= 1:
                count = 2
                j = i + 1
                while j < n and nums[j] - nums[i] <= j - i:
                    count += 1
                    j += 1
                max_count = max(max_count, count)
                i = j - 1
        
        return max_count