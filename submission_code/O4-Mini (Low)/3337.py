class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        # Count how many times character c appears in s
        k = s.count(c)
        # Every substring that starts and ends with c corresponds
        # to choosing a start position (one of the k) and an end position
        # at or after it (another of the k, possibly the same).
        # That yields k*(k+1)/2 substrings.
        return k * (k + 1) // 2