class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        max_length = 0
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                consecutive_pair_count = 0
                for k in range(len(sub) - 1):
                    if sub[k] == sub[k+1]:
                        consecutive_pair_count += 1
                if consecutive_pair_count <= 1:
                    max_length = max(max_length, len(sub))
        return max_length