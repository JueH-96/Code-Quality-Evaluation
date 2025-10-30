class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        max_outlier = float('-inf')
        
        for num in nums:
            # Calculate the sum of special numbers by excluding the current number
            special_sum = total_sum - num
            
            # Check if the current number is a valid outlier
            if special_sum in nums:
                max_outlier = max(max_outlier, num)
        
        return max_outlier