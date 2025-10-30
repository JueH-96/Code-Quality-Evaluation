class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        # Count frequency of each number
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # Get list of frequencies
        counts = list(freq.values())
        min_freq = min(counts)
        
        # Try each possible group size from min_freq down to 1
        for size in range(min_freq, 0, -1):
            valid = True
            total_groups = 0
            
            # For each frequency, check if it can be divided into groups
            # where difference between number of elements in groups is at most 1
            for count in counts:
                # Get number of groups needed for this frequency
                groups = count // size
                remainder = count % size
                
                # If remainder exists, need one more group
                if remainder > 0:
                    groups += 1
                    # Check if this is valid - remainder should be >= groups-1
                    if remainder < groups - 1:
                        valid = False
                        break
                
                total_groups += groups
            
            if valid:
                return total_groups
        
        return len(nums)