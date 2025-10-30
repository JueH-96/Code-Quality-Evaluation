class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        min_sum = float('inf')
        found = False
        
        # For each possible subarray length from l to r
        for size in range(l, r + 1):
            # Calculate sum for first window
            curr_sum = sum(nums[:size])
            
            # If sum is greater than 0, update min_sum
            if curr_sum > 0:
                min_sum = min(min_sum, curr_sum)
                found = True
            
            # Slide the window through the array
            for i in range(size, n):
                # Add new element and remove first element of previous window
                curr_sum = curr_sum + nums[i] - nums[i - size]
                
                # If sum is greater than 0, update min_sum
                if curr_sum > 0:
                    min_sum = min(min_sum, curr_sum)
                    found = True
        
        # If no valid subarray found, return -1
        return min_sum if found else -1