class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        shortest_len = float('inf')
        beautiful_substrings = []

        for i in range(n):
            ones_count = 0
            for j in range(i, n):
                if s[j] == '1':
                    ones_count += 1
                if ones_count == k:
                    beautiful_substrings.append(s[i:j+1])
                    shortest_len = min(shortest_len, j - i + 1)
                    break

        if not beautiful_substrings:
            return ""

        shortest_substrings = [s for s in beautiful_substrings if len(s) == shortest_len]
        shortest_substrings.sort()

        return shortest_substrings[0]