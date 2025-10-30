class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        left_sum = 0
        count = 0
        
        # Iterate through the array to calculate left and right sums
        for i in range(len(nums) - 1):
            left_sum += nums[i]
            right_sum = total_sum - left_sum
            
            # Check if the difference is even
            if (left_sum - right_sum) % 2 == 0:
                count += 1
        
        return count