from collections import Counter

class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        count = Counter(s)
        if not count:
            return ""
        max_freq = max(count.values())
        result = []
        current_counts = {}
        for char in s:
            current_counts[char] = current_counts.get(char, 0) + 1
            if current_counts[char] == max_freq:
                result.append(char)
        return ''.join(result)