class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        """
        We say a (contiguous) substring has "dominant ones" when the 
        number of 1s in that substring (call it c1) satisfies:

            c1 >= (number_of_zeros_in_substring)^2

        Equivalently, if a substring has c0 zeros, then it must have
        at least c0^2 ones. Since any substring length is c0 + c1,
        we need c1 >= c0^2 => (c0 + c1) - c0 >= c0^2 => length >= c0^2 + c0.

        Also observe that if c0 > sqrt(n), then c0^2 > n, and no substring
        of length at most n can have c1 >= c0^2. Hence we only need to
        consider c0 up to floor(sqrt(n)).

        To count efficiently:
          1) Let n = len(s). Gather the indices of all '0's in a list zpos.
             Suppose there are K zeros. We create an extended list Z:
                 Z = [-1] + zpos + [n]
          2) Substrings with no zeros (c0=0) are just runs of consecutive '1's.
             Each run of length L contributes L*(L+1)//2 such substrings.
          3) For c0 in [1.. min(K, floor(sqrt(n)))] we:
               - Let L0 = c0^2 + c0.
               - We consider exactly c0 zeros at positions Z[i]..Z[i+c0-1].
               - Such a substring must start in [Z[i-1]+1 .. Z[i]] 
                 (to include Z[i] but not Z[i-1]) 
                 and end   in [Z[i+c0-1] .. Z[i+c0]-1] 
                 (to include Z[i+c0-1] but not Z[i+c0]).
               - Denote these intervals as A..B for starts, C..D for ends.
               - Among all possible (start, end) in that rectangle, we only
                 count those whose length >= L0.
               - We derive an O(1) closed-form for counting such pairs.
          4) Summing up yields the total number of substrings with dominant ones.

        This solution runs in O(n * sqrt(n)) in the worst case but is optimized
        enough to pass large constraints in Python if implemented carefully.
        """

        s_len = len(s)
        zeros = []
        for i, ch in enumerate(s):
            if ch == '0':
                zeros.append(i)
        K = len(zeros)

        # Extended zero positions with sentinel boundaries
        Z = [-1] + zeros + [s_len]

        # 1) Count substrings with no zeros (c0=0): runs of '1's
        total = 0
        for i in range(1, K + 1):
            # The run of 1's is between Z[i-1]+1 and Z[i]-1.
            run_len = Z[i] - Z[i - 1] - 1
            if run_len > 0:
                total += run_len * (run_len + 1) // 2

        # A helper to sum integers from u to v inclusive in O(1).
        def intsum(u, v):
            # sum(u..v) = v*(v+1)//2 - (u*(u-1))//2
            # but written carefully to handle edge cases:
            if v < u:
                return 0
            return (v * (v + 1) // 2) - ((u - 1) * u // 2)

        from math import isqrt
        max_c0 = isqrt(s_len)  # we only need to check up to sqrt(n)

        # 2) For c0 in [1.. min(K, max_c0)], count substrings with exactly c0 zeros
        #    whose length >= c0^2 + c0
        for c0 in range(1, min(K, max_c0) + 1):
            # length needed = c0^2 + c0
            L0 = c0 * c0 + c0
            # Slide over blocks of exactly c0 zeros: Z[i]..Z[i+c0-1]
            # valid starts in [Z[i-1]+1 .. Z[i]] and
            # valid ends   in [Z[i+c0-1] .. Z[i+c0] - 1]
            for i in range(K - c0 + 1):
                A = Z[i - 1] + 1
                B = Z[i]
                C = Z[i + c0 - 1]
                D = Z[i + c0] - 1
                if A > B or C > D:
                    continue

                # Max possible length in this block is (D - A + 1). If smaller than L0, skip.
                if (D - A + 1) < L0:
                    continue

                # Number of ways to pick a start in [A..B] is x
                x = (B - A + 1)
                # We want all (s,e) with s in [A..B], e in [C..D], e - s + 1 >= L0
                # i.e. e >= s + (L0-1). We do a piecewise sum:
                eS = A + (L0 - 1)       # minimal e if we fix s = A
                Estart = max(eS, C)     # e cannot be less than C
                Eend2 = min(D, B + L0 - 1)  # beyond B+(L0-1), #s(e) saturates at x

                # Sum for e in [Estart..Eend2]: #s(e) = (e - A - (L0 - 2))
                # because s can go up to e-(L0-1). Then we handle the "saturates" region:
                # for e in [Eend2+1..D], we have x possible starts if that region is valid.

                sum2 = 0
                if Estart <= Eend2:
                    length2 = Eend2 - Estart + 1
                    # sum of e in [Estart..Eend2]
                    s_e = intsum(Estart, Eend2)
                    # each term is (e - A - (L0 - 2)) = e - (A + L0 - 2)
                    sum2 = s_e - (A + L0 - 2) * length2

                sum3 = 0
                if Eend2 < D:
                    # e in (Eend2+1..D] => each e gives x starts
                    sum3 = x * (D - Eend2)

                total += (sum2 + sum3)

        return total