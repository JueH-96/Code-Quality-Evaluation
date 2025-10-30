class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total = 0
        for start in range(n):
            counts = {}
            max_freq = 0
            for end in range(start, n):
                char = s[end]
                counts[char] = counts.get(char, 0) + 1
                if counts[char] > max_freq:
                    max_freq = counts[char]
                if max_freq >= k:
                    total += (n - end)
                    break  # No need to check further for this start
        return total