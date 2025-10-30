class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        min_length = float('inf')
        candidates = []
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                substr = s[i:j+1]
                count = substr.count('1')
                if count == k:
                    current_length = j - i + 1
                    if current_length < min_length:
                        min_length = current_length
                        candidates = [substr]
                    elif current_length == min_length:
                        candidates.append(substr)
        if not candidates:
            return ""
        return min(candidates)