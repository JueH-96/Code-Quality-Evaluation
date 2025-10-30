class Solution:
    def minimumSum(self, nums) -> int:
        min_sum = float('inf')
        n = len(nums)
        for j in range(1, n - 1):
            # Find the minimum left element smaller than nums[j]
            left = [nums[i] for i in range(j) if nums[i] < nums[j]]
            if not left:
                continue  # No valid left element
            
            # Find the minimum right element smaller than nums[j]
            right = [nums[k] for k in range(j + 1, n) if nums[k] < nums[j]]
            if not right:
                continue  # No valid right element
            
            current_sum = min(left) + nums[j] + min(right)
            if current_sum < min_sum:
                min_sum = current_sum
        
        return min_sum if min_sum != float('inf') else -1