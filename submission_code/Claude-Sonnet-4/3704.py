class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = sum(nums)
        count = 0
        left_sum = 0
        
        # Try each possible partition index i (0 to n-2)
        for i in range(n - 1):
            left_sum += nums[i]
            right_sum = total_sum - left_sum
            
            # Check if difference is even
            diff = left_sum - right_sum
            if diff % 2 == 0:
                count += 1
        
        return count