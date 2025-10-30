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
        
        # Try to find the maximum possible group size k such that all frequencies can be divided into groups of size k or k+1
        # We need to find the largest k <= min_freq such that for all f in freq_list, f can be divided into groups of size k or k+1
        
        # We will try k from min_freq down to 1
        for k in range(min_freq, 0, -1):
            valid = True
            for f in freq_list:
                # Calculate the number of groups needed for frequency f with group sizes k and k+1
                # We need to find x and y such that x * k + y * (k+1) = f, and x + y is minimized
                # To minimize the number of groups, we maximize the number of groups of size k+1
                # So y = f // (k+1)
                # x = (f - y * (k+1)) // k
                y = f // (k + 1)
                x = (f - y * (k + 1)) // k
                if x * k + y * (k + 1) != f:
                    valid = False
                    break
            if valid:
                # Calculate the total number of groups
                total_groups = 0
                for f in freq_list:
                    y = f // (k + 1)
                    x = (f - y * (k + 1)) // k
                    total_groups += x + y
                return total_groups
        
        # If no k is found, return the sum of frequencies (each element in its own group)
        return sum(freq_list)