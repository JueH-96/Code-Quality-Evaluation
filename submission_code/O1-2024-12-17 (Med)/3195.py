class Solution:
    def minimumSteps(self, s: str) -> int:
        # We count the total number of steps needed to move all '1's (black) 
        # to the right and all '0's (white) to the left.
        # This is effectively the number of inversions (i < j, s[i] = '1', s[j] = '0'), 
        # because each inversion requires exactly one adjacent swap to fix.

        steps = 0
        count_ones = 0  # To keep track of how many '1's we've seen so far.
        
        for char in s:
            if char == '1':
                # Increase the count of ones encountered
                count_ones += 1
            else:
                # For each '0', add the number of ones that appear before it
                steps += count_ones

        return steps