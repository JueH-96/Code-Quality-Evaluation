class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        n = len(nums)
        special_sum = 0
        special_numbers = set()
        
        # Find the sum of the special numbers and store them in a set
        for i in range(n):
            if i == 0 or i == n-1 or nums[i] not in special_numbers:
                special_sum += nums[i]
                special_numbers.add(nums[i])
        
        # Find the largest outlier
        outlier = None
        for num in nums:
            if num != special_sum - num and num not in special_numbers:
                if outlier is None or num > outlier:
                    outlier = num
        
        return outlier