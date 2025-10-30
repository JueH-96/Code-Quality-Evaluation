class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # For each possible partition point
        for i in range(n - 1):
            # Calculate sum of left subarray [0, i]
            left_sum = sum(nums[:i + 1])
            # Calculate sum of right subarray [i + 1, n-1]
            right_sum = sum(nums[i + 1:])
            # Check if difference is even
            if (left_sum - right_sum) % 2 == 0:
                count += 1
                
        return count