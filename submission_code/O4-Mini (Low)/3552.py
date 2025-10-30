import math

class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # Number of mirrored positions (left half)
        L = n // 2
        # Precompute powers of 10 modulo k up to n
        pow10 = [1] * (n)
        for i in range(1, n):
            pow10[i] = (pow10[i-1] * 10) % k

        # Compute weights for each mirrored position i in [0..L-1]:
        # w[i] = (10^i + 10^(n-1-i)) mod k
        w = [(pow10[i] + pow10[n-1-i]) % k for i in range(L)]
        # If n is odd, compute the middle weight
        if n % 2 == 1:
            w_mid = pow10[L]  # weight for the center digit
        else:
            w_mid = 0

        # Compute G0 = gcd of all nonzero weights and k.
        # If all weights are zero mod k, set G0 = 0 as a sentinel.
        G0 = 0
        for x in w:
            if x != 0:
                G0 = x if G0 == 0 else math.gcd(G0, x)
        if w_mid != 0:
            G0 = w_mid if G0 == 0 else math.gcd(G0, w_mid)
        if G0 != 0:
            G0 = math.gcd(G0, k)
        # If no nonzero weights, all palindromes are divisible by k automatically
        # mark with G0 = 0
        if G0 == k:
            # Then effectively G0=0 as well (all residues ok)
            G0 = 0

        # Build the left half greedily
        left = []
        rem = 0  # current remainder mod k
        for i in range(L):
            wi = w[i]
            # choose digit d from 9 down to 0 (or 1 for i=0)
            low = 1 if i == 0 else 0
            for d in range(9, low-1, -1):
                nr = (rem + d * wi) % k
                # check if from here we can finish to a multiple of k:
                # if G0==0, always possible; else nr % G0 must be 0.
                if G0 == 0 or nr % G0 == 0:
                    left.append(str(d))
                    rem = nr
                    break
            else:
                # theoretically should not happen under constraints
                left.append('0')
                rem = (rem + 0 * wi) % k

        # Middle digit if odd
        mid_char = ''
        if n % 2 == 1:
            # choose d from 9..0 so that (rem + d*w_mid) % k == 0
            for d in range(9, -1, -1):
                if (rem + d * w_mid) % k == 0:
                    mid_char = str(d)
                    rem = (rem + d * w_mid) % k
                    break

        # Build right half as mirror of left
        right = ''.join(reversed(left))
        return ''.join(left) + mid_char + right