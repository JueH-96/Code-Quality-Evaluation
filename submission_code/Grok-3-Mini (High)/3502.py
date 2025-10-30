class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total = n * (n + 1) // 2
        freq = [0] * 26
        left = 0
        bad_count = 0
        for right in range(n):
            char_idx = ord(s[right]) - ord('a')
            freq[char_idx] += 1
            while max(freq) >= k:
                left_char_idx = ord(s[left]) - ord('a')
                freq[left_char_idx] -= 1
                left += 1
            bad_count += (right - left + 1)
        return total - bad_count