class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        n = len(s)
        ans = 0
        # Try all possible starting points
        for i in range(n):
            freq = {}
            # Extend substring from i to j
            for j in range(i, n):
                freq[s[j]] = freq.get(s[j], 0) + 1
                # If any character occurs more than twice, stop extending
                if freq[s[j]] > 2:
                    break
                ans = max(ans, j - i + 1)
        return ans