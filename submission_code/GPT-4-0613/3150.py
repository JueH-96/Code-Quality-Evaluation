class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        beautiful_substrings = []
        for i in range(n):
            for j in range(i + k, n + 1):
                if s[i:j].count('1') == k:
                    beautiful_substrings.append(s[i:j])
        if not beautiful_substrings:
            return ""
        min_len = min(len(x) for x in beautiful_substrings)
        beautiful_substrings = [x for x in beautiful_substrings if len(x) == min_len]
        return min(beautiful_substrings)