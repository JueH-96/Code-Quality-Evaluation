class Solution:
    def minimumSteps(self, s: str) -> int:
        # The goal is to move all '0's to the left and '1's to the right.
        # The minimum number of adjacent swaps to achieve this is the total
        # number of inversions, where an inversion is a '1' that appears
        # before a '0'.

        # We can count these inversions in a single pass.
        # We iterate through the string, keeping a count of the '1's seen so far.
        # When we encounter a '0', it must be swapped with all the '1's
        # that have appeared to its left. The number of such swaps is
        # equal to the current count of '1's.
        
        swaps = 0
        ones_count = 0
        
        for character in s:
            if character == '1':
                ones_count += 1
            else:  # character == '0'
                # This '0' forms an inversion with each of the `ones_count` '1's
                # seen so far. We add this to the total swaps.
                swaps += ones_count
                
        return swaps