class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        binaryPowerOf5 = set((1 << i) for i in range(7))

        @cache
        def fn(i):
            """Return minimum number of beautiful strings."""
            if i == len(s): 
                return 0
            if s[i] == "0": 
                return inf 
            ans = inf
            for j in range(i, len(s)): 
                x = int(s[i:j+1], 2)
                if x in binaryPowerOf5: 
                    ans = min(ans, 1 + fn(j+1))
            return ans 

        ans = fn(0) 
        return ans if ans < inf else -1