class Solution:
    def maxSum(self, nums: List[int]) -> int:
        n = len(nums)
        max_sum = float('-inf')
        
        # Try all possible subarrays
        for i in range(n):
            seen = set()
            current_sum = 0
            
            for j in range(i, n):
                # If we see a duplicate, we would delete it
                if nums[j] not in seen:
                    seen.add(nums[j])
                    current_sum += nums[j]
                    max_sum = max(max_sum, current_sum)
        
        return max_sum