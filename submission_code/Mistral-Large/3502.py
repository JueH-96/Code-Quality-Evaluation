class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        from collections import defaultdict

        n = len(s)
        count = 0

        for i in range(n):
            char_count = defaultdict(int)
            max_freq = 0
            for j in range(i, n):
                char_count[s[j]] += 1
                max_freq = max(max_freq, char_count[s[j]])
                if max_freq >= k:
                    count += 1

        return count