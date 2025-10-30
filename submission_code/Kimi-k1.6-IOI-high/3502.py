class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        total = len(s) * (len(s) + 1) // 2
        invalid = self.count_invalid(s, k)
        return total - invalid
    
    def count_invalid(self, s: str, k: int) -> int:
        n = len(s)
        count = [0] * 26
        left = 0
        result = 0
        for right in range(n):
            idx = ord(s[right]) - ord('a')
            count[idx] += 1
            while count[idx] >= k:
                left_idx = ord(s[left]) - ord('a')
                count[left_idx] -= 1
                left += 1
            result += (right - left + 1)
        return result