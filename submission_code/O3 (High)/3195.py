class Solution:
    def minimumSteps(self, s: str) -> int:
        """
        The task reduces to counting how many times a black ball ('1') appears
        to the left of a white ball ('0').  Each such (1,0) pair is an inversion
        that must be resolved by exactly one adjacent swap, so the total number
        of inversions is the minimum number of required swaps.

        We can count these inversions in a single pass from left to right:
        - keep a running count of how many '1's have been seen so far
        - whenever we see a '0', it forms an inversion with every '1' before it,
          so we add the current number of seen '1's to the answer.
        """
        ones_so_far = 0  # number of '1's encountered to the left
        steps = 0        # total swaps (inversions)

        for ch in s:
            if ch == '1':
                ones_so_far += 1
            else:  # ch == '0'
                steps += ones_so_far

        return steps