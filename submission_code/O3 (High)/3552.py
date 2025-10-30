class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # When k == 1 every integer is divisible by 1, so the largest
        # n–digit palindrome is just n copies of '9'.
        if k == 1:
            return '9' * n

        mod = k

        # ------------------------------------------------------------------
        # 1.  Pre–compute 10^p (mod k) for every position (right to left)
        # ------------------------------------------------------------------
        pow10_mod = [0] * n
        pow10_mod[-1] = 1 % mod                 # 10^0 (mod k)
        for i in range(n - 2, -1, -1):
            pow10_mod[i] = (pow10_mod[i + 1] * 10) % mod

        # ------------------------------------------------------------------
        # 2.  Build the list of “groups”.
        #     Each group corresponds to one free digit in the palindrome:
        #       – The first  ⌊n/2⌋ groups are symmetric pairs
        #       – One extra group is added when n is odd (the middle digit)
        # ------------------------------------------------------------------
        half = n // 2
        groups_weights = []
        for i in range(half):                             # paired positions
            w = (pow10_mod[i] + pow10_mod[n - 1 - i]) % mod
            groups_weights.append(w)

        if n & 1:                                         # centre digit
            groups_weights.append(pow10_mod[half])

        L = len(groups_weights)                           # number of groups

        # ------------------------------------------------------------------
        # 3.  Suffix–feasibility DP.
        #     suff[i] is a bitmask: bit r is 1  ⇔  remainder r can be achieved
        #     using groups  i, i+1, … , L-1  (with any digits 0–9).
        # ------------------------------------------------------------------
        suff = [0] * (L + 1)
        suff[L] = 1                                       # only remainder 0 after the end

        for i in range(L - 1, -1, -1):
            w = groups_weights[i] % mod
            nxt_mask = suff[i + 1]
            mask = 0
            for r in range(mod):
                if (nxt_mask >> r) & 1:                   # suffix can give remainder r
                    for d in range(10):                   # try all digits for this group
                        new_r = (r + d * w) % mod
                        mask |= 1 << new_r
            suff[i] = mask

        # ------------------------------------------------------------------
        # 4.  Greedy construction (left → right).
        #     Always pick the largest possible digit that still allows a full
        #     completion to remainder 0, using the pre–computed suff array.
        # ------------------------------------------------------------------
        prefix_rem = 0
        left_digits = []          # digits of the left half (as integers)
        middle_digit = None

        for i in range(L):
            w = groups_weights[i] % mod

            # Digit range: the very first group cannot be 0 (no leading zeros)
            digit_iter = range(9, 0, -1) if i == 0 else range(9, -1, -1)

            chosen = None
            for d in digit_iter:
                rem_after = (prefix_rem + d * w) % mod
                need = (-rem_after) % mod                # what the suffix must produce
                if (suff[i + 1] >> need) & 1:            # feasible?
                    chosen = d
                    prefix_rem = rem_after
                    break

            # In valid inputs a solution always exists; the check below is
            # only for completeness.
            if chosen is None:
                return ""

            # Store the chosen digit
            if i < half:
                left_digits.append(chosen)
            else:                                        # middle group (only when n is odd)
                middle_digit = chosen

        # ------------------------------------------------------------------
        # 5.  Assemble the palindrome string
        # ------------------------------------------------------------------
        left_part = ''.join(str(d) for d in left_digits)
        if n & 1:
            palindrome = left_part + str(middle_digit) + left_part[::-1]
        else:
            palindrome = left_part + left_part[::-1]

        return palindrome