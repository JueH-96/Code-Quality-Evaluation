class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        count = Counter(s)
        result = 0
        for char, freq in count.items():
            result += (freq + 1) // 2
        return result