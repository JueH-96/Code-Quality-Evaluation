class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all valid powers of 5 (in binary) that fit within 15 bits
        powers = []
        val = 1
        while True:
            bin_str = bin(val)[2:]  # binary representation (without '0b')
            if len(bin_str) > 15:   # no need to go further if binary length exceeds 15
                break
            powers.append(bin_str)
            val *= 5
        
        # Use memoization to avoid recomputing results
        from functools import lru_cache
        
        @lru_cache(None)
        def dfs(idx):
            # If we've reached the end, no more substrings are needed
            if idx == len(s):
                return 0
            
            best = float('inf')
            # Try matching any power-of-5 substring at position idx
            for p in powers:
                # Check if s[idx:] starts with this power-of-5 string p
                if s.startswith(p, idx):
                    res = dfs(idx + len(p))
                    if res != float('inf'):
                        best = min(best, 1 + res)
            
            return best
        
        ans = dfs(0)
        return ans if ans != float('inf') else -1