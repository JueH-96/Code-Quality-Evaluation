class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # Compute 10^i mod k for i = 0..n-1
        pow10 = [1] * n
        for i in range(1, n):
            pow10[i] = (pow10[i-1] * 10) % k

        # Determine half-length h and split index L
        L = n // 2
        r = n % 2
        h = L + r

        # Compute weight w_i for each position i in [0..h-1]
        w = [0] * h
        if r == 1:
            # odd length: central index is L
            for i in range(L):
                w[i] = (pow10[i] + pow10[n-1-i]) % k
            w[L] = pow10[L] % k
        else:
            # even length: no center digit
            for i in range(h):
                w[i] = (pow10[i] + pow10[n-1-i]) % k

        # We don't need pow10 any more
        del pow10

        # DP_mask[i] is a bitmask of remainders mod k possible from suffix i..h-1
        # bit r is 1 if remainder r is achievable
        full_mask = (1 << k) - 1
        DP_mask = [0] * (h + 1)
        DP_mask[h] = 1  # only remainder 0 is possible with empty suffix

        # Build DP_mask backwards
        for i in range(h-1, -1, -1):
            mask_next = DP_mask[i+1]
            wi = w[i]
            mask_i = 0
            # Allowed digits at position i: cannot have leading zero at i=0
            if i == 0:
                d_start, d_end = 1, 9
            else:
                d_start, d_end = 0, 9

            # If weight is zero, every digit contributes zero shift
            if wi == 0:
                # any allowed digit yields the same mask_next
                # if there is at least one allowed digit, mask_i = mask_next
                mask_i = mask_next
            else:
                # Otherwise accumulate shifts for each digit
                for d in range(d_start, d_end+1):
                    t = (d * wi) % k
                    if t == 0:
                        mask_i |= mask_next
                    else:
                        # cyclic shift by t bits in k-bit mask
                        shifted = ((mask_next << t) | (mask_next >> (k - t))) & full_mask
                        mask_i |= shifted
            DP_mask[i] = mask_i

        # Now reconstruct the digits greedily for the largest palindrome
        # We want total remainder 0, so start with curr_r = 0
        curr_r = 0
        a = [0] * h
        for i in range(0, h):
            wi = w[i]
            if i == 0:
                d_start, d_end = 1, 9
            else:
                d_start, d_end = 0, 9
            # Try digits from largest to smallest
            for d in range(d_end, d_start-1, -1):
                t = (d * wi) % k
                prev_r = (curr_r - t) % k
                # Check if suffix i+1..h-1 can achieve prev_r
                if (DP_mask[i+1] >> prev_r) & 1:
                    a[i] = d
                    curr_r = prev_r
                    break

        # Build the palindrome string from the first half (and middle if odd)
        first_half = ''.join(str(d) for d in a[:L])
        if r == 1:
            middle = str(a[L])
            second_half = first_half[::-1]
            return first_half + middle + second_half
        else:
            second_half = first_half[::-1]
            return first_half + second_half