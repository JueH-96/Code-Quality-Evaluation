class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total_substrings = n * (n + 1) // 2
        if k <= 0:
            return total_substrings
        
        freq = [0] * 26
        left = 0
        count = 0
        total_invalid = 0  # Substrings where no char appears at least k times
        
        for right in range(n):
            index = ord(s[right]) - ord('a')
            freq[index] += 1
            if freq[index] == k:
                count += 1
            while count > 0:
                left_index = ord(s[left]) - ord('a')
                if freq[left_index] == k:
                    count -= 1
                freq[left_index] -= 1
                left += 1
            total_invalid += right - left + 1
        
        return total_substrings - total_invalid