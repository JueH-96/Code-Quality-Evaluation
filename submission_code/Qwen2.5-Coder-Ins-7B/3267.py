class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        max_length = -1
        
        for i in range(n):
            for j in range(i + 1, n + 1):
                substring = s[i:j]
                if len(set(substring)) == 1:
                    count = s.count(substring)
                    if count >= 3:
                        max_length = max(max_length, len(substring))
        
        return max_length