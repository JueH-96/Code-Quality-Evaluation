class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        len_s = len(s)
        len_p = len(pattern)
        
        for i in range(len_s - len_p + 1):
            substring = s[i:i + len_p]
            diff_count = sum(1 for a, b in zip(substring, pattern) if a != b)
            if diff_count <= 1:
                return i
        
        return -1