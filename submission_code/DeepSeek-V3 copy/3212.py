class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        # Create a dictionary to store the last occurrence of each number
        last_occurrence = {}
        for idx, num in enumerate(nums):
            last_occurrence[num] = idx
        
        # Initialize the end of the current segment
        end = 0
        count = 1
        
        for idx, num in enumerate(nums):
            # Update the end of the current segment
            end = max(end, last_occurrence[num])
            # If the current index is the end of the segment, it's a possible partition point
            if idx == end:
                count *= 2
                count %= (10**9 + 7)
        
        # Since we have counted all possible partitions, but the last segment is always included
        # We need to divide by 2 to get the correct count
        return count // 2 % (10**9 + 7)