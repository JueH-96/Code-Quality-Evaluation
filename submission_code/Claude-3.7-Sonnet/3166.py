from math import ceil

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        # Count frequencies of each value
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        # Get all frequencies
        frequencies = list(count.values())
        min_freq = min(frequencies)
        
        # Try all possible values of group size from min_freq down to 1
        for size in range(min_freq, 0, -1):
            valid = True
            
            for freq in frequencies:
                # Check if this frequency can be validly split
                # into groups of size 'size' and 'size+1'
                if (freq // size) - (freq % size) < 0:
                    valid = False
                    break
            
            if valid:
                # Calculate total number of groups needed
                total_groups = 0
                for freq in frequencies:
                    total_groups += ceil(freq / (size + 1))
                return total_groups
        
        # This should never happen if the input is valid
        return len(nums)