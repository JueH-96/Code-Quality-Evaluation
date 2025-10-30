class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find first and last occurrence of each number
        first_occurrence = {}
        last_occurrence = {}
        
        for i, num in enumerate(nums):
            if num not in first_occurrence:
                first_occurrence[num] = i
            last_occurrence[num] = i
        
        # Create intervals for each unique number
        intervals = []
        for num in first_occurrence:
            intervals.append((first_occurrence[num], last_occurrence[num]))
        
        # Sort intervals by start position
        intervals.sort()
        
        # Merge overlapping intervals
        merged = []
        for start, end in intervals:
            if not merged or merged[-1][1] < start:
                # No overlap, add new interval
                merged.append([start, end])
            else:
                # Overlap, extend the last interval
                merged[-1][1] = max(merged[-1][1], end)
        
        # Number of good partitions is 2^(number of merged intervals - 1)
        # Because between each pair of consecutive intervals, we can choose to split or not
        num_segments = len(merged)
        
        # Calculate 2^(num_segments - 1) mod MOD
        if num_segments == 0:
            return 1
        
        return pow(2, num_segments - 1, MOD)