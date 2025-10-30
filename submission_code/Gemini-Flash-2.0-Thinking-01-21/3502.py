class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        for i in range(n):
            freq_map = {}
            for j in range(i, n):
                char = s[j]
                freq_map[char] = freq_map.get(char, 0) + 1
                for char_count in freq_map.values():
                    if char_count >= k:
                        count += 1
                        break
        return count