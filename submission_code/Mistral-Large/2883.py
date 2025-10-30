class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        powers_of_5 = set()
        x = 1
        while x <= int('1' * n, 2):
            powers_of_5.add(bin(x)[2:])
            x *= 5

        def is_beautiful(substring):
            return substring in powers_of_5

        def min_partitions(start):
            if start == n:
                return 0
            if s[start] == '0':
                return float('inf')
            min_count = float('inf')
            for end in range(start + 1, n + 1):
                if is_beautiful(s[start:end]):
                    min_count = min(min_count, 1 + min_partitions(end))
            return min_count

        result = min_partitions(0)
        return result if result != float('inf') else -1