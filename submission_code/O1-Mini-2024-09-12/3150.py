class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        candidate = []
        min_length = n + 1
        for i in range(n):
            count = 0
            for j in range(i, n):
                if s[j] == '1':
                    count += 1
                if count == k:
                    current_len = j - i + 1
                    if current_len < min_length:
                        min_length = current_len
                        candidate = [s[i:j+1]]
                    elif current_len == min_length:
                        candidate.append(s[i:j+1])
                    break
                elif count > k:
                    break
        if not candidate:
            return ""
        return min(candidate)