class Solution:
    def minimumSteps(self, s: str) -> int:
        # The minimum number of adjacent swaps needed to
        # group all '1's to the right (and '0's to the left)
        # is simply the inversion count of the binary string:
        # count of pairs (i, j) with i < j, s[i]=='1', s[j]=='0'.
        
        ones = 0
        swaps = 0
        for ch in s:
            if ch == '1':
                # we saw another '1'
                ones += 1
            else:
                # ch == '0', each prior '1' must swap past this '0'
                swaps += ones
        return swaps