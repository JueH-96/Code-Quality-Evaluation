class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Sort the array to make it easier to process
        nums.sort()
        
        # For each number, we want to try to make it different from its neighbors
        # by using the allowed operations [-k, k]
        result = 1  # First number is always distinct
        last_used = nums[0]  # Keep track of the last number we used
        
        for i in range(1, n):
            # Current number we're trying to place
            curr = nums[i]
            
            # If current number is already different from last used number
            if curr > last_used:
                result += 1
                last_used = curr
                continue
                
            # Try to make current number different by adding up to k
            # We need curr + x > last_used where -k <= x <= k
            # So x > last_used - curr
            needed_diff = last_used - curr + 1
            
            if needed_diff <= k:
                # We can make this number different
                result += 1
                last_used = curr + needed_diff
            
        return result