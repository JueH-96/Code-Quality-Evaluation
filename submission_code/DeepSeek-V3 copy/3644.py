class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        min_sum = float('inf')
        
        # Iterate over all possible starting positions
        for i in range(n):
            current_sum = 0
            # Iterate over all possible lengths from l to r
            for length in range(l, r+1):
                if i + length > n:
                    break
                # Calculate the sum of the subarray from i to i+length-1
                current_sum = sum(nums[i:i+length])
                if current_sum > 0 and current_sum < min_sum:
                    min_sum = current_sum
        
        return min_sum if min_sum != float('inf') else -1