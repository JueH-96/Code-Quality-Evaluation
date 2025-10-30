class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def count_non_zero_upto(n):
            s = str(n)
            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dfs(pos, tight, leading_zero):
                if pos == len(s):
                    return 0 if leading_zero else 1
                res = 0
                max_d = int(s[pos]) if tight else 9
                for d in range(0, max_d + 1):
                    new_tight = tight and (d == max_d)
                    new_leading = leading_zero and (d == 0)
                    if new_leading:
                        res += dfs(pos + 1, new_tight, new_leading)
                    else:
                        if d == 0:
                            continue  # Skip zero if not leading
                        res += dfs(pos + 1, new_tight, False)
                return res

            return dfs(0, True, True)

        # Calculate count_non_zero_in_range(l, r)
        count_non_zero_r = count_non_zero_upto(r)
        count_non_zero_l_minus_1 = count_non_zero_upto(l - 1)
        count_non_zero_in_range = count_non_zero_r - count_non_zero_l_minus_1

        # Compute count_with_zero
        total_numbers = r - l + 1
        count_with_zero = total_numbers - count_non_zero_in_range

        # Now compute count_non_zero_but_satisfy_condition
        def generate_non_zero_numbers(l, r):
            from collections import deque
            queue = deque()
            # Initialize with single-digit numbers
            for d in range(1, 10):
                num = d
                if num >= l and num <= r:
                    yield num
                if num < r:
                    queue.append(num)
            # Generate multi-digit numbers
            while queue:
                current = queue.popleft()
                for d in range(1, 10):
                    next_num = current * 10 + d
                    if next_num > r:
                        continue
                    if next_num >= l:
                        yield next_num
                    if next_num < r:
                        queue.append(next_num)

        count = 0
        for num in generate_non_zero_numbers(l, r):
            s = 0
            p = 1
            n = num
            while n > 0:
                d = n % 10
                s += d
                p *= d
                n = n // 10
            if p % s == 0:
                count += 1

        return count_with_zero + count