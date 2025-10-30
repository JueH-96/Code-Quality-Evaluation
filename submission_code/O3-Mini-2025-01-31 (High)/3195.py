class Solution:
    def minimumSteps(self, s: str) -> int:
        # We'll count the number of inversions in the string,
        # where an inversion is a pair (i, j) with i < j, s[i]=='1' and s[j]=='0'.
        # The number of such inversions is the minimum number of adjacent swaps needed.
        
        one_count = 0  # count of '1's so far (black balls)
        steps = 0      # count of swaps/inversions
        
        for char in s:
            if char == '1':
                one_count += 1
            else:  # char is '0'
                steps += one_count
                
        return steps