from functools import lru_cache

class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        # Count beautiful numbers in [l, r] = f(r) - f(l-1)
        return self._count_up_to(r) - self._count_up_to(l - 1)

    def _count_up_to(self, N: int) -> int:
        if N <= 0:
            return 0
        digs = list(map(int, str(N)))
        n = len(digs)

        # 1) Count numbers <= N that contain no zero digit
        @lru_cache(None)
        def dp_no_zero(pos, tight, started):
            if pos == n:
                # valid if we've started (non-empty number)
                return 1 if started else 0
            limit = digs[pos] if tight else 9
            res = 0
            for d in range(0, limit + 1):
                nt = tight and (d == limit)
                if not started:
                    # still skipping leading zeros
                    if d == 0:
                        res += dp_no_zero(pos + 1, nt, False)
                    else:
                        # start with non-zero digit, and ensure it's not zero => ok
                        res += dp_no_zero(pos + 1, nt, True)
                else:
                    # already started, we must not pick digit zero
                    if d == 0:
                        continue
                    res += dp_no_zero(pos + 1, nt, True)
            return res

        total = N
        no_zero = dp_no_zero(0, True, False)
        have_zero = total - no_zero  # all numbers that have at least one zero are beautiful

        # 2) Count numbers <= N with no zero digit AND product % sum == 0
        good_nonzero = 0
        # We try all possible sum of digits s from 1..9*len
        max_sum = 9 * n
        for target_sum in range(1, max_sum + 1):
            @lru_cache(None)
            def dp2(pos, tight, started, sum_so_far, prod_mod):
                # sum_so_far <= target_sum always
                if sum_so_far > target_sum:
                    return 0
                if pos == n:
                    # valid if started, sum matches, and prod_mod == 0
                    return 1 if (started and sum_so_far == target_sum and prod_mod == 0) else 0
                limit = digs[pos] if tight else 9
                res = 0
                for d in range(0, limit + 1):
                    nt = tight and (d == limit)
                    if not started:
                        # leading zeros allowed, don't count towards sum/product
                        if d == 0:
                            res += dp2(pos + 1, nt, False, 0, 1 % target_sum)
                        else:
                            # start here with non-zero, initialize product=d
                            if d <= target_sum:
                                res += dp2(pos + 1, nt, True,
                                           d,
                                           d % target_sum)
                    else:
                        # already started => digits 1..9 only (no zeros)
                        if d == 0:
                            continue
                        new_sum = sum_so_far + d
                        if new_sum > target_sum:
                            continue
                        new_mod = (prod_mod * d) % target_sum
                        res += dp2(pos + 1, nt, True, new_sum, new_mod)
                return res

            # If target_sum=0 it doesn't make sense; skip.
            # Init prod_mod with 1 % target_sum for leading-zero state
            # Only call dp2 if target_sum>0
            good_nonzero += dp2(0, True, False, 0, 1 % target_sum)

            dp2.cache_clear()

        return have_zero + good_nonzero