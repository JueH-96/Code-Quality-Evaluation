from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        """
        We need to pick a subsequence of length 2*k from nums in its original order. 
        Split that subsequence into two halves (each of length k) and compute:
            (OR of first half) XOR (OR of second half)
        We want to maximize that value.

        Approach:
         1) Let n = len(nums).
         2) We use dynamic programming to find all possible OR-values of picking j elements
            from the first i elements (prefix), and similarly from the subarray i..n-1 (suffix).
         
         3) Define fwd[i][j] as a bitmask where the b-th bit is set if there is a way
            to pick j elements from nums[:i] (i.e., first i elements) such that their OR equals b.
             - We'll have fwd dimension: (n+1) x (k+1), each an integer bitmask up to 128 bits 
               (since nums[i] < 2^7, the OR can be at most 127).
             - Base cases: fwd[0][0] = 1<<0 (OR=0 is possible with 0 picked), else 0.
             - Recurrence:
                  fwd[i][j] = fwd[i-1][j]  (not pick nums[i-1])
                               OR shiftbits(fwd[i-1][j-1], nums[i-1]) (pick nums[i-1])
         
         4) Similarly define bwd[i][j] as a bitmask for picking j elements from nums[i:], i.e. from index i onward.
             - We'll fill it backward from i = n down to 0.
             - Base cases: bwd[n][0] = 1<<0, else 0.
             - Recurrence:
                  bwd[i][j] = bwd[i+1][j] (skip nums[i])
                              OR shiftbits(bwd[i+1][j-1], nums[i]) (pick nums[i])
         
         5) Then, to form a valid subsequence of length 2*k, we pick k elements entirely from [0..i], 
            and k elements from [i..n]. The boundary i ensures the first half's indices are all < i, 
            and the second half's indices are all >= i, thus preserving order with no overlap.
            So for i in [k..n-k], we combine all possible OR-values from fwd[i][k] with those from bwd[i][k].
            The max over (or1 XOR or2) is our answer.

         6) Implementation detail:
             - We'll create a helper to "shift" a bitmask by OR-ing with a given value: 
               for every set bit b in the mask, we set bit (b|val) in the new mask.
             - Then do DP forward (fwd) and backward (bwd).
             - Finally, iterate over i in [k..n-k], decode each bitmask into a list of OR-values, 
               compute the best XOR pair, track the global maximum.

        Time complexity:
         - Building fwd and bwd: O(n * k * 128).
         - Combining results: O(n * 128^2) in the worst case.
         - Feasible for n <= 400, k <= 200, and values < 128.
        """

        n = len(nums)
        # Precompute the operation "x|val" for x in [0..127] and val in [0..127] 
        # to speed up the bitmask shifting.
        # shift_table[x][v] = (x | v).
        shift_table = [[0]*128 for _ in range(128)]
        for x in range(128):
            for v in range(128):
                shift_table[x][v] = x | v

        # A helper to set bits for (each b set in mask) OR val
        def shiftbits(mask: int, val: int) -> int:
            # We'll iterate over bits set in mask:
            # new_mask will have bits set at (b | val) for each b in mask.
            # Where b is the integer index of the bit (0..127).
            new_mask = 0
            m = mask
            while m:
                # get the lowest set bit
                low = m & -m
                bit_index = low.bit_length() - 1
                # set the bit (bit_index | val)
                new_mask |= 1 << shift_table[bit_index][val]
                # clear that bit
                m ^= low
            return new_mask

        # fwd[i][j] : bitmask of possible OR-values picking j elements from nums[:i]
        fwd = [[0]*(k+1) for _ in range(n+1)]
        fwd[0][0] = 1 << 0  # OR = 0 is possible with 0 elements

        for i in range(1, n+1):
            num = nums[i-1]
            for j in range(k+1):
                # not pick
                fwd[i][j] = fwd[i-1][j]
                # pick
                if j > 0:
                    fwd[i][j] |= shiftbits(fwd[i-1][j-1], num)

        # bwd[i][j]: bitmask of possible OR-values picking j elements from nums[i:]
        bwd = [[0]*(k+1) for _ in range(n+1)]
        bwd[n][0] = 1 << 0  # picking 0 elements => OR=0

        for i in range(n-1, -1, -1):
            num = nums[i]
            for j in range(k+1):
                # skip
                bwd[i][j] = bwd[i+1][j]
                # pick
                if j > 0:
                    bwd[i][j] |= shiftbits(bwd[i+1][j-1], num)

        # Now combine: for each boundary i in [k..n-k]
        # we combine the OR-values from fwd[i][k] with bwd[i][k].
        best = 0
        for i in range(k, n - k + 1):
            left_mask = fwd[i][k]
            right_mask = bwd[i][k]
            if left_mask == 0 or right_mask == 0:
                continue

            # Decode left_mask bits
            left_vals = []
            temp = left_mask
            while temp:
                bit = temp & -temp
                idx = bit.bit_length() - 1
                left_vals.append(idx)
                temp ^= bit

            # Decode right_mask bits
            right_vals = []
            temp = right_mask
            while temp:
                bit = temp & -temp
                idx = bit.bit_length() - 1
                right_vals.append(idx)
                temp ^= bit

            # For each possible or1, find the best XOR with elements in right_vals
            # We'll do a simple nested loop (at most 128^2 per boundary).
            # Keep track of maximum.
            for or1 in left_vals:
                for or2 in right_vals:
                    candidate = or1 ^ or2
                    if candidate > best:
                        best = candidate

        return best