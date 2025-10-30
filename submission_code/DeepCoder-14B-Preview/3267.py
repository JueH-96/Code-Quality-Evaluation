from collections import defaultdict

class Solution:
    def maximumLength(self, s: str) -> int:
        counts = defaultdict(int)
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                substr = s[i:j+1]
                if all(c == substr[0] for c in substr):
                    counts[substr] += 1
        max_len = -1
        for substr, cnt in counts.items():
            if cnt >= 3:
                current_len = len(substr)
                if current_len > max_len:
                    max_len = current_len
        return max_len if max_len != -1 else -1