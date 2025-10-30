from collections import Counter

class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        cnt = Counter(s)
        max_freq = max(cnt.values())
        max_chars = {char for char, freq in cnt.items() if freq == max_freq}
        last_occurrence = {}
        for idx, char in enumerate(s):
            if char in max_chars:
                last_occurrence[char] = idx
        sorted_chars = sorted(last_occurrence.keys(), key=lambda x: last_occurrence[x])
        return ''.join(sorted_chars)