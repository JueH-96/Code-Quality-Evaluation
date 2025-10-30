from math import factorial
from functools import lru_cache

class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        """
        Count integers in [l, r] whose product of digits is divisible
        by the sum of digits.
        """

        # ------------------ pre–computation ------------------
        fact = [1] * 10
        for i in range(1, 10):
            fact[i] = fact[i - 1] * i

        # convenience: permutations of a multiset (list of 10 counters)
        def perm(cnt):
            total = sum(cnt)
            res = fact[total]
            for c in cnt:
                res //= fact[c]
            return res

        # permutations with first (most–significant) digit ≠ 0
        def perm_first_non_zero(cnt):
            if cnt[0] == 0:
                return perm(cnt)
            # fix a leading zero and count the remaining permutations
            cnt[0] -= 1
            res_leading_zero = perm(cnt)
            cnt[0] += 1
            return perm(cnt) - res_leading_zero

        good_multisets_by_len = [[] for _ in range(10)]   #  index = length (1 … 9)
        total_good_by_len     = [0] * 10                  #  total permutations (first digit ≠ 0)

        # generate every digit multiset with at most 9 digits
        cnt = [0] * 10

        def dfs(digit: int, used: int):
            """enumerate counters; digit = current digit (0‥9), used = digits chosen so far"""
            if digit == 10:
                if used == 0:           # skip empty multiset
                    return
                # check “beautiful”
                if cnt[0] > 0:          # any zero ⇒ product = 0 ⇒ automatically beautiful
                    beautiful = True
                else:
                    s = sum(i * cnt[i] for i in range(1, 10))
                    prod = 1
                    for i in range(1, 10):
                        prod *= i ** cnt[i]
                    beautiful = (prod % s == 0)

                if beautiful:
                    first_nz = perm_first_non_zero(cnt)
                    if first_nz:        # ignore multisets that cannot start with non–zero (e.g. only zeros)
                        tup = tuple(cnt)
                        good_multisets_by_len[used].append(tup)
                        total_good_by_len[used] += first_nz
                return

            max_take = 9 - used
            for k in range(max_take + 1):
                cnt[digit] = k
                dfs(digit + 1, used + k)
            cnt[digit] = 0

        dfs(0, 0)        # fill good_multisets_by_len & totals

        # ----------------------------------------------------
        # helper: permutations of 'cnt' not exceeding the decimal string 'limit'
        # ----------------------------------------------------
        @lru_cache(maxsize=None)
        def perm_cached(t):
            """cached version of perm"""
            return perm(list(t))

        def count_perms_up_to(limit: str, counter_tuple):
            """How many permutations (first digit ≠ 0) of this multiset form numbers ≤ limit?"""
            cnt = list(counter_tuple)
            n = len(limit)
            ans = 0
            for pos in range(n):
                curr_digit = int(limit[pos])
                start = 1 if pos == 0 else 0
                for d in range(start, curr_digit):
                    if cnt[d] == 0:
                        continue
                    cnt[d] -= 1
                    ans += perm_cached(tuple(cnt))
                    cnt[d] += 1
                if cnt[curr_digit] == 0:        # cannot place current digit, stop
                    return ans
                cnt[curr_digit] -= 1
            # if all digits consumed, the number itself is counted
            return ans + 1

        # ----------------------------------------------------
        def beautiful_up_to(N: int) -> int:
            """number of beautiful integers in [1, N]"""
            if N <= 0:
                return 0
            s = str(N)
            L = len(s)
            total = 0

            # 1) all lengths strictly smaller than L
            for d in range(1, L):
                total += total_good_by_len[d]

            # 2) length == L : need to respect the upper bound
            for multiset in good_multisets_by_len[L]:
                total += count_perms_up_to(s, multiset)

            return total

        # ----------------------------------------------------
        return beautiful_up_to(r) - beautiful_up_to(l - 1)