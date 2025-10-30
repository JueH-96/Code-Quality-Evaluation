class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        def count_substrings_with_at_most_k_chars(s, k, char):
            count = 0
            left = 0
            for right in range(len(s)):
                if s[right] == char:
                    k -= 1
                while k < 0:
                    if s[left] == char:
                        k += 1
                    left += 1
                count += right - left + 1
            return count
        
        total_substrings = len(s) * (len(s) + 1) // 2
        invalid_substrings_0 = count_substrings_with_at_most_k_chars(s, -k-1, '0')
        invalid_substrings_1 = count_substrings_with_at_most_k_chars(s, -k-1, '1')
        
        return total_substrings - invalid_substrings_0 - invalid_substrings_1