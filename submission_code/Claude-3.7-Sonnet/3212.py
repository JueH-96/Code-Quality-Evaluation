class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        # Find the last occurrence of each number
        last_occurrence = {}
        for i, num in enumerate(nums):
            last_occurrence[num] = i
        
        # Count the minimum number of required partitions
        partitions = 0
        max_end = 0
        
        for i, num in enumerate(nums):
            # Update the furthest endpoint we need to include in current partition
            max_end = max(max_end, last_occurrence[num])
            
            # If we've reached the end of a required partition
            if i == max_end:
                partitions += 1
        
        # Calculate total ways: 2^(partitions-1)
        # For each partition boundary (except the last), we can choose to split or not
        MOD = 10**9 + 7
        return pow(2, partitions - 1, MOD)