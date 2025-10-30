class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        shortest_beautiful_substrings = []

        for i in range(n):
            ones_count = 0
            for j in range(i, n):
                if s[j] == '1':
                    ones_count += 1

                if ones_count == k:
                    substring = s[i:j+1]
                    current_len = len(substring)
                    if current_len < min_len:
                        min_len = current_len
                        shortest_beautiful_substrings = [substring]
                    elif current_len == min_len:
                        shortest_beautiful_substrings.append(substring)

        if not shortest_beautiful_substrings:
            return ""
        else:
            return min(shortest_beautiful_substrings)