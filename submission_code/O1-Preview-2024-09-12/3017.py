class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        from functools import lru_cache

        def count_beautiful_numbers_upto(limit, k):
            if limit <= 0:
                return 0
            s = str(limit)
            n = len(s)

            from functools import lru_cache

            MAX_DIFF = n  # Maximum possible count difference between even and odd digits

            @lru_cache(None)
            def dfs(pos, diff, rem, tight, started):
                if pos == n:
                    if started and diff == 0 and rem == 0:
                        return 1
                    else:
                        return 0
                count = 0
                max_digit = int(s[pos]) if tight else 9
                for digit in range(0, max_digit +1):
                    new_tight = tight and (digit == max_digit)
                    new_started = started or digit != 0
                    new_diff = diff
                    new_rem = rem
                    if new_started:
                        if digit % 2 == 0:
                            new_diff +=1
                        else:
                            new_diff -=1
                        new_rem = (rem *10 + digit) % k
                    else:
                        # If number hasn't started yet, diff and rem remain the same
                        pass
                    count += dfs(pos+1, new_diff, new_rem, new_tight, new_started)
                return count

            return dfs(0, 0, 0, True, False)

        return count_beautiful_numbers_upto(high, k) - count_beautiful_numbers_upto(low -1, k)