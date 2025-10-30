from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        # We only need to track for each intermediate string:
        # its first character, its last character, and its length.
        # dp[f][l] = minimum length achievable for a string whose
        # first char is f and last char is l (0 <= f,l < 26).
        INF = 10**18
        dp = [[INF] * 26 for _ in range(26)]
        
        # Initialize with the first word
        f0 = ord(words[0][0]) - ord('a')
        l0 = ord(words[0][-1]) - ord('a')
        dp[f0][l0] = len(words[0])
        
        # Process each subsequent word
        for w in words[1:]:
            w0 = ord(w[0]) - ord('a')
            w1 = ord(w[-1]) - ord('a')
            wl = len(w)
            
            new_dp = [[INF] * 26 for _ in range(26)]
            
            # Try extending each possible (first, last) state
            for f in range(26):
                for l in range(26):
                    cur_len = dp[f][l]
                    if cur_len == INF:
                        continue
                    
                    # Option 1: join(previous_string, w)
                    # Drops one character if l == w0
                    cost1 = cur_len + wl - (1 if l == w0 else 0)
                    # Resulting string has first = f, last = w1
                    if cost1 < new_dp[f][w1]:
                        new_dp[f][w1] = cost1
                    
                    # Option 2: join(w, previous_string)
                    # Drops one character if w1 == f
                    cost2 = cur_len + wl - (1 if w1 == f else 0)
                    # Resulting string has first = w0, last = l
                    if cost2 < new_dp[w0][l]:
                        new_dp[w0][l] = cost2
            
            dp = new_dp
        
        # The answer is the minimum length over all (first, last) pairs
        ans = min(dp[f][l] for f in range(26) for l in range(26))
        return ans