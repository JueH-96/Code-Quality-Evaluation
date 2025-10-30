class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        start = 0
        max_size = 0
        last_seen = {}
        for i, c in enumerate(s):
            last_seen[c] = last_seen.get(c, 0) + 1
            if last_seen[c] > 2:
                max_size = max(max_size, i - start)
                while last_seen[c] > 2:
                    last_seen[s[start]] -= 1
                    if last_seen[s[start]] == 0:
                        last_seen.pop(s[start])
                    start += 1
        max_size = max(max_size, len(s) - start)
        return max_size