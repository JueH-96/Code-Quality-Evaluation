from collections import Counter

class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        count = Counter(s)
        max_freq = max(count.values())
        max_chars = [c for c in count if count[c] == max_freq]
        indices = [s.rfind(c) for c in max_chars]
        indices.sort()
        return ''.join(s[i] for i in indices)