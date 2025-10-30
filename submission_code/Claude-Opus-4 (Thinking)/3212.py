class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find last occurrence of each element
        last = {}
        for i, num in enumerate(nums):
            last[num] = i
        
        # Count the number of gaps between mandatory groups
        gaps = 0
        max_reach = 0
        
        for i, num in enumerate(nums):
            # Update the furthest index we need to reach
            max_reach = max(max_reach, last[num])
            
            # If we've reached the end of a mandatory group (and not at the end)
            if i == max_reach and i < len(nums) - 1:
                gaps += 1
        
        # Number of partitions is 2^gaps
        return pow(2, gaps, MOD)