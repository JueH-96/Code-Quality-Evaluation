class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find first and last occurrence of each unique number
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
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        # Count partitions: 2^(number of merged intervals - 1)
        return pow(2, len(merged) - 1, MOD)