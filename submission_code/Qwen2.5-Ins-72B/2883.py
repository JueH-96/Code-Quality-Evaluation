class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        powers_of_5 = {bin(5**i)[2:] for i in range(7)}  # Precompute powers of 5 in binary form
        n = len(s)
        
        def is_beautiful(sub):
            return sub in powers_of_5
        
        def dfs(start, count):
            if start == n:
                return count
            if s[start] == '0':
                return float('inf')
            min_count = float('inf')
            for end in range(start + 1, n + 1):
                if is_beautiful(s[start:end]):
                    min_count = min(min_count, dfs(end, count + 1))
            return min_count
        
        result = dfs(0, 0)
        return result if result != float('inf') else -1