class Solution:
    def maximumLength(self, s: str) -> int:
        counts = {}
        for char in set(s):
            counts[char] = s.count(char)
        
        max_len = -1
        for length in range(1, len(s) + 1):
            for i in range(len(s) - length + 1):
                substring = s[i:i+length]
                if len(set(substring)) == 1 and counts[substring[0]] >= 3:
                    max_len = max(max_len, length)
        return max_len