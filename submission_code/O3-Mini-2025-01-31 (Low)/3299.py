from typing import List
from collections import Counter

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = Counter(nums)
        # For each number as a candidate base x, try building chain:
        # Chain rule: base = a0, then a1 = a0^2, a2 = a1^2, ... etc.
        # The pattern is: 
        # if the chain extends m steps (m>=1), the full array is:
        # [a0, a1, ..., a_m, ..., a1, a0]
        # which requires for i in [0, m-1]: freq[a_i] >= 2 and for a_m: freq[a_m] >= 1.
        # If chain length m=0 then it is just [a0] (requires freq[a0]>=1).
        # We'll compute for every candidate a0 in the freq dictionary
        # and update the maximum chain length (number of elements in the palindrome)

        res = 1  # at least one element can be chosen
        
        # We iterate over each candidate that occurs in our array
        for x in freq:
            # m = 0 chain always possible if freq[x]>=1, chain length = 1.
            best = 1
            # If we want to extend chain beyond base, need freq[x]>=2
            if freq[x] < 2:
                # no extension possible, so chain remains [x]
                res = max(res, best)
                continue

            current = x
            m = 0
            # We try to extend the chain while possible.
            # For each extension, let next_num = current^2
            # For extension i-th (starting count from 0): for the current "layer" we require current's
            # frequency to be at least 2. Then we compute next number.
            while True:
                # For extension, first check if the current layer (base for next step) is available in pair.
                if m == 0:
                    # already checked for base: freq[x]>=2 (because we're in the extension branch)
                    pass
                else:
                    # for layers beyond base we already ensured they have been used in pair in previous iteration,
                    # so nothing extra here.
                    pass

                # Compute next candidate as square of current; note: use multiplication instead of exponentiation.
                try:
                    next_val = current * current
                except OverflowError:
                    break
                # next_val must exist in our frequency dictionary.
                # For the extension to be valid beyond this, we require:
                # freq[current] (for the previous layer) was already used in pair, and now:
                # For the new candidate level, we need at least 1 occurrence if we decide to end here,
                # BUT if we want to further extend, we need it appear at least 2.
                if next_val not in freq:
                    # Cannot extend the chain further: so we can end the chain at current level if not extended.
                    # But note: the chain was built up to m, and now cannot extend to m+1.
                    break

                # We can extend one more layer.
                # But to use next_val in the middle of a palindrome (i.e. as an inner pair
                # rather than the center), we need at least 2 occurrence.
                if freq[next_val] < 2:
                    # We can use next_val as the center part (which only requires 1).
                    # So current chain extension: previously extended m times, now ending with next_val as center.
                    # Chain length becomes: 2*(m+1) + 1, because we have a0,...,a_m, a_{m+1}, a_m,...,a0.
                    best = max(best, 2*(m+1) + 1)
                    break
                else:
                    # We can extend and use next_val as paired element.
                    m += 1
                    # Update best chain length so far if we decide to stop at next iteration (by using next_val as center)
                    best = max(best, 2*m + 1)
                    current = next_val
            res = max(res, best)
        return res