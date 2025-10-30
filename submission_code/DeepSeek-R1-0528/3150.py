class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        res = ""
        for start in range(n):
            ones_count = 0
            for end in range(start, n):
                if s[end] == '1':
                    ones_count += 1
                if ones_count == k:
                    candidate = s[start:end+1]
                    candidate_len = end - start + 1
                    if candidate_len < min_len:
                        min_len = candidate_len
                        res = candidate
                    elif candidate_len == min_len:
                        if candidate < res:
                            res = candidate
                    break
        return res