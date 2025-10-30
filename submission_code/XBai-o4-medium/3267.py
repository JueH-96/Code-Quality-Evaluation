from collections import defaultdict

class Solution:
    def maximumLength(self, s: str) -> int:
        if not s:
            return -1
        groups = []
        prev = s[0]
        count = 1
        for c in s[1:]:
            if c == prev:
                count += 1
            else:
                groups.append((prev, count))
                prev = c
                count = 1
        groups.append((prev, count))
        
        freq = defaultdict(int)
        for char, l in groups:
            for k in range(1, l + 1):
                contribution = l - k + 1
                freq[(char, k)] += contribution
        
        max_len = -1
        for (char, length), total in freq.items():
            if total >= 3:
                if length > max_len:
                    max_len = length
        return max_len if max_len != -1 else -1