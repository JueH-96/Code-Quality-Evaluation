class Solution:
    def minimizedStringLength(self, s: str) -> int:
        count = [0]*26
        for char in s:
            count[ord(char) - ord('a')] += 1
        return len(set(s))