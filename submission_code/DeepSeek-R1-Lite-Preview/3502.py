class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total_substrings = n * (n + 1) // 2
        
        if k == 1:
            return total_substrings
        
        freq = {}
        left = 0
        invalid_char_count = 0
        count_invalid_substrings = 0
        
        for right in range(n):
            char = s[right]
            freq[char] = freq.get(char, 0) + 1
            if freq[char] == k:
                invalid_char_count += 1
            
            while invalid_char_count > 0:
                left_char = s[left]
                freq[left_char] -= 1
                if freq[left_char] == k - 1:
                    invalid_char_count -= 1
                left += 1
            
            count_invalid_substrings += right - left + 1
        
        return total_substrings - count_invalid_substrings