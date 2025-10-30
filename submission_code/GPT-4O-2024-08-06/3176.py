class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        
        # Iterate over each possible peak index j
        for j in range(1, n - 1):
            # Find the best i < j such that nums[i] < nums[j]
            min_left = float('inf')
            for i in range(j):
                if nums[i] < nums[j]:
                    min_left = min(min_left, nums[i])
            
            # Find the best k > j such that nums[k] < nums[j]
            min_right = float('inf')
            for k in range(j + 1, n):
                if nums[k] < nums[j]:
                    min_right = min(min_right, nums[k])
            
            # If both sides have valid elements, calculate the sum
            if min_left != float('inf') and min_right != float('inf'):
                current_sum = min_left + nums[j] + min_right
                min_sum = min(min_sum, current_sum)
        
        # If no valid mountain triplet was found, return -1
        return min_sum if min_sum != float('inf') else -1