class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        """
        Enumerate all palindromes having exactly n digits.
        Each palindrome that is divisible by k certifies its multiset of
        digits as “good”.  For every such multiset we count how many
        different n–digit integers (with no leading zero) can be formed
        from it and add the value once (duplicates are avoided with a
        dictionary).
        Because n ≤ 10 the amount of work is tiny:
            total palindromes = 9·10^{⌈n/2⌉−1} ≤ 90 000
        """
        # ------------- small helpers -------------
        # factorials 0 … n
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i

        def permutations_without_leading_zero(cnt):
            """
            number of distinct permutations of the multiset *cnt*
            that produce an n–digit number (no leading zero)
            """
            # denominator common to all permutations
            base = 1
            for c in cnt:
                base *= fact[c]

            total = 0
            f = fact[n - 1]          # (n-1)!
            for d in range(1, 10):   # leading digit must be non-zero
                c = cnt[d]
                if c == 0:
                    continue
                #  (n-1)! * c / base            (see derivation in analysis)
                total += (f * c) // base
            return total

        # ------------- enumerate palindromes -------------
        half_len = (n + 1) // 2                # length of the left half (incl. middle digit if n odd)
        first_half_start = 1 if half_len == 1 else 10 ** (half_len - 1)
        first_half_end   = 10 ** half_len      # exclusive upper bound

        seen_multisets = {}   # multiset (tuple of counts) -> permutations count
        answer = 0

        for left in range(first_half_start, first_half_end):
            left_str = str(left).zfill(half_len)   # ensure proper length
            if n & 1:                              # odd length
                right_str = left_str[:-1][::-1]
            else:                                  # even length
                right_str = left_str[::-1]
            full_str = left_str + right_str

            # divisibility test
            if int(full_str) % k:
                continue

            # build digit multiset
            cnt = [0] * 10
            for ch in full_str:
                cnt[ord(ch) - 48] += 1
            key = tuple(cnt)

            if key not in seen_multisets:
                seen_multisets[key] = permutations_without_leading_zero(cnt)
                answer += seen_multisets[key]

        return answer