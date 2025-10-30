class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        max_diff = 0
        n = len(nums)
        
        for i in range(n):
            # Calculate difference with next element (circular using modulo)
            next_index = (i + 1) % n
            diff = abs(nums[i] - nums[next_index])
            max_diff = max(max_diff, diff)
        
        return max_diff