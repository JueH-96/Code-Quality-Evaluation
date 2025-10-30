class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        def is_power_of_five(n):
            while n % 5 == 0:
                n //= 5
            return n == 1
        
        def dfs(start):
            if start == len(s):
                return 0
            min_partitions = float('inf')
            for end in range(start + 1, len(s) + 1):
                if s[start] == '0':
                    continue
                num = int(s[start:end], 2)
                if is_power_of_five(num):
                    partitions = 1 + dfs(end)
                    min_partitions = min(min_partitions, partitions)
            return min_partitions if min_partitions != float('inf') else -1
        
        return dfs(0)