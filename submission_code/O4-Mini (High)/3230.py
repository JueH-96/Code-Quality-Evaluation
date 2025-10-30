class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        # DP where dp_prev[c] = minimum changes up to previous position, ending with char c
        n = len(word)
        # Map characters to 0..25
        orig = [ord(ch) - ord('a') for ch in word]
        
        # Initialize dp for position 0
        INF = 10**9
        dp_prev = [INF] * 26
        for c in range(26):
            dp_prev[c] = 0 if c == orig[0] else 1
        
        # Fill DP for positions 1..n-1
        for i in range(1, n):
            dp_curr = [INF] * 26
            for c in range(26):
                # cost to change position i to char c
                cost = 0 if c == orig[i] else 1
                # find best previous char p with |p-c| >= 2
                best = INF
                for p in range(26):
                    if abs(p - c) >= 2:
                        if dp_prev[p] < best:
                            best = dp_prev[p]
                dp_curr[c] = best + cost
            dp_prev = dp_curr
        
        # Answer is min cost ending with any char at last position
        return min(dp_prev)