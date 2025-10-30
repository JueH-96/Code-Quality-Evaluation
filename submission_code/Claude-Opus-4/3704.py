class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        
        # If total sum is odd, no partition can have even difference
        if total_sum % 2 == 1:
            return 0
        
        # Count partitions where left_sum is even
        count = 0
        left_sum = 0
        
        # Try each partition point (0 to n-2)
        for i in range(len(nums) - 1):
            left_sum += nums[i]
            if left_sum % 2 == 0:
                count += 1
        
        return count