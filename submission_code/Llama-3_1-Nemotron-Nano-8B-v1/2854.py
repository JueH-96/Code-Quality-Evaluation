from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        if not words:
            return 0
        
        # Precompute first and last characters and lengths for each word
        first_last = []
        for word in words:
            f = word[0]
            l = word[-1]
            first_last.append((f, l, len(word)))
        
        # Initialize DP with the first word's state
        dp = {}
        f0, l0, _ = first_last[0]
        dp[(f0, l0)] = len(words[0])
        
        for i in range(1, len(words)):
            c, d, m = first_last[i]
            new_dp = {}
            # Iterate through all possible previous states
            for (a_prev, b_prev), prev_len in dp.items():
                # Option 1: join previous string with current word
                new_a = a_prev
                new_b = d
                overlap = 1 if b_prev == c else 0
                new_len = prev_len + m - overlap
                key = (new_a, new_b)
                if key not in new_dp or new_len < new_dp[key]:
                    new_dp[key] = new_len
                
                # Option 2: join current word with previous string
                new_a2 = c
                new_b2 = b_prev
                overlap2 = 1 if d == a_prev else 0
                new_len2 = m + prev_len - overlap2
                key2 = (new_a2, new_b2)
                if key2 not in new_dp or new_len2 < new_dp[key2]:
                    new_dp[key2] = new_len2
            dp = new_dp
        
        return min(dp.values()) if dp else 0