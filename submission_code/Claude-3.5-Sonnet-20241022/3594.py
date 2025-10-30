class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        n = len(nums)
        # For each number, check if it could be the outlier by removing it
        # and checking if the remaining numbers can satisfy the condition
        max_outlier = float('-inf')
        
        for i in range(n):
            # Try current number as outlier
            remaining = nums[:i] + nums[i+1:]
            # For each potential sum in remaining numbers
            for j in range(len(remaining)):
                potential_sum = remaining[j]
                # Remove this potential sum
                numbers = remaining[:j] + remaining[j+1:]
                # Check if numbers sum up to potential_sum
                curr_sum = sum(numbers)
                if curr_sum == potential_sum:
                    # nums[i] is a valid outlier
                    max_outlier = max(max_outlier, nums[i])
                    break
        
        return max_outlier