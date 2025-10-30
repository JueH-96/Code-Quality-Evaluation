class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        if n <= 1:
            return 0
        
        # Convert letters to 0..25
        s = [ord(c) - ord('a') for c in word]
        INF = 10**9
        
        # dp_prev[c]: min changes up to prev position, ending with char c
        dp_prev = [INF] * 26
        # Base case for position 0
        for c in range(26):
            dp_prev[c] = 0 if c == s[0] else 1
        
        # Iterate positions 1..n-1
        for i in range(1, n):
            # Precompute prefix and suffix minima of dp_prev
            prefix_min = [0] * 26
            suffix_min = [0] * 26
            m = INF
            for c in range(26):
                m = min(m, dp_prev[c])
                prefix_min[c] = m
            m = INF
            for c in range(25, -1, -1):
                m = min(m, dp_prev[c])
                suffix_min[c] = m
            
            dp = [INF] * 26
            for c in range(26):
                # We need prev char pc with |c - pc| > 1
                best = INF
                # prev in [0..c-2]
                if c - 2 >= 0:
                    best = min(best, prefix_min[c-2])
                # prev in [c+2..25]
                if c + 2 < 26:
                    best = min(best, suffix_min[c+2])
                # Cost to change current to c
                cost = 0 if c == s[i] else 1
                dp[c] = best + cost
            dp_prev = dp
        
        # Answer is minimum over last position endings
        return min(dp_prev)