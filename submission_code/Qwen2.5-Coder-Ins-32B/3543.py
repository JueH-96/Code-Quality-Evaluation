class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        def count_substrings_with_max_char(s, max_char, k):
            count = 0
            left = 0
            char_count = 0
            
            for right in range(len(s)):
                if s[right] == max_char:
                    char_count += 1
                
                while char_count > k:
                    if s[left] == max_char:
                        char_count -= 1
                    left += 1
                
                count += right - left + 1
            
            return count
        
        count_0s = count_substrings_with_max_char(s, '0', k)
        count_1s = count_substrings_with_max_char(s, '1', k)
        
        # Subtract the overcounted substrings where both 0's and 1's are at most k
        count_both = count_substrings_with_max_char(s, '0', k) + count_substrings_with_max_char(s, '1', k) - len(s) * (len(s) + 1) // 2
        
        return count_0s + count_1s - count_both