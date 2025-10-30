class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        from functools import lru_cache

        max_num = 2 ** 15  # maximum number possible with 15 bits
        # Precompute powers of 5 up to max_num
        powers_of_5 = set()
        power = 1
        while power <= max_num:
            powers_of_5.add(power)
            power *= 5

        n = len(s)

        @lru_cache(maxsize=None)
        def dp(i):
            if i == n:
                return 0  # no more characters to process
            if s[i] == '0':
                return float('inf')  # cannot start with '0'
            min_substrings = float('inf')
            num = 0
            for j in range(i, n):
                num = num * 2 + int(s[j])
                if num in powers_of_5:
                    next_substrings = dp(j + 1)
                    if next_substrings != float('inf'):
                        min_substrings = min(min_substrings, 1 + next_substrings)
            return min_substrings

        result = dp(0)
        return result if result != float('inf') else -1