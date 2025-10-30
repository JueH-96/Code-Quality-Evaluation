class Solution:
    def minimumSteps(self, s: str) -> int:
        """
        We need to group all '0's to the left and all '1's to the right.
        The minimum number of adjacent swaps required to sort a binary string
        is exactly the number of inversions, i.e., the number of pairs (i<j)
        with s[i] = '1' and s[j] = '0'.

        We can count these inversions in one pass:
        - Keep a running count of '1's seen so far.
        - Every time we see a '0', it contributes that many inversions.
        """
        ones = 0
        inversions = 0
        for ch in s:
            if ch == '1':
                ones += 1
            else:  # ch == '0'
                inversions += ones
        return inversions