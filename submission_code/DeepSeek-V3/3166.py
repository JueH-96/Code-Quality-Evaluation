from collections import defaultdict

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        # Count the frequency of each number
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        # Get the list of frequencies
        freq_list = list(freq.values())
        
        # Find the minimum frequency
        min_freq = min(freq_list)
        
        # Try to find the maximum possible group size
        # We need to find the largest k such that all frequencies can be divided into groups of size k or k+1
        # We can try k from min_freq down to 1
        for k in range(min_freq, 0, -1):
            possible = True
            total_groups = 0
            for f in freq_list:
                # Calculate the number of groups needed for this frequency
                # We need to divide f into groups of size k or k+1
                # The number of groups is ceil(f / (k+1)) but also must be at least ceil(f / (k+1))
                # To minimize the number of groups, we try to use as many k+1 as possible
                # So, the number of groups is ceil(f / (k+1)) but also must be such that f >= groups * k and f <= groups * (k+1)
                # So, groups = ceil(f / (k+1)) and f >= groups * k
                groups = (f + k) // (k + 1)
                if groups * k > f:
                    possible = False
                    break
                total_groups += groups
            if possible:
                return total_groups
        
        return len(freq_list)