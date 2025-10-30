class Solution:
    def minimumSteps(self, s: str) -> int:
        # We will count how many '1's we have encountered so far.
        # Each time we encounter a '0', we add the number of '1's found so far to our answer.
        # This effectively counts how many times a '1' appears before a '0' (i.e., "10" inversions).
        
        count_ones = 0
        steps = 0
        
        for char in s:
            if char == '1':
                count_ones += 1
            else:  # char == '0'
                # Each '0' adds as many swaps as the number of '1's before it
                steps += count_ones
        
        return steps