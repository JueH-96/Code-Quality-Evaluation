class Solution:
    def minimumSteps(self, s: str) -> int:
        # We need to sort the string so that all '0's come before all '1's,
        # using adjacent swaps. The minimum number of adjacent swaps needed
        # is exactly the inversion count of pairs (i, j) where i < j, s[i] = '1', s[j] = '0'.
        # We can compute this by counting how many '1's we've seen so far when
        # we encounter a '0'.
        
        ones_count = 0
        swaps = 0
        for ch in s:
            if ch == '1':
                ones_count += 1
            else:  # ch == '0'
                swaps += ones_count
        return swaps