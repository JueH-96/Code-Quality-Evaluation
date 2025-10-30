class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        def count_substrings_with_max_char(s, char, k):
            count = 0
            left = 0
            max_count = 0
            
            for right in range(len(s)):
                if s[right] == char:
                    count += 1
                
                while count > k:
                    if s[left] == char:
                        count -= 1
                    left += 1
                
                max_count += right - left + 1
            
            return max_count
        
        return count_substrings_with_max_char(s, '0', k) + count_substrings_with_max_char(s, '1', k)