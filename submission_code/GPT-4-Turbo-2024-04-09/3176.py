class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return -1
        
        # Initialize the minimum sum to a large number
        min_sum = float('inf')
        
        # Iterate over each possible peak of the mountain
        for j in range(1, n-1):
            left_min = float('inf')
            right_min = float('inf')
            
            # Find the minimum element on the left of j that is less than nums[j]
            for i in range(j):
                if nums[i] < nums[j]:
                    left_min = min(left_min, nums[i])
            
            # Find the minimum element on the right of j that is less than nums[j]
            for k in range(j+1, n):
                if nums[k] < nums[j]:
                    right_min = min(right_min, nums[k])
            
            # If valid left and right mins are found, calculate the sum
            if left_min != float('inf') and right_min != float('inf'):
                current_sum = left_min + nums[j] + right_min
                min_sum = min(min_sum, current_sum)
        
        # If no valid mountain was found, return -1
        if min_sum == float('inf'):
            return -1
        else:
            return min_sum