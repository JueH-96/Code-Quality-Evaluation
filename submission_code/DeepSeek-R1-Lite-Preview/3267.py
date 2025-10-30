class Solution:
    def maximumLength(self, s: str) -> int:
        max_length = -1
        n = len(s)
        for length in range(n, 0, -1):
            for i in range(n - length + 1):
                substr = s[i:i+length]
                if len(set(substr)) == 1:
                    count = s.count(substr)
                    if count >= 3:
                        return length
        return max_length