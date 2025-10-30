class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        def max_substring_length(s, k, char):
            left = 0
            max_len = 0
            count = 0
            
            for right in range(len(s)):
                if s[right] != char:
                    count += 1
                
                while count > k:
                    if s[left] != char:
                        count -= 1
                    left += 1
                
                max_len = max(max_len, right - left + 1)
            
            return max_len
        
        max_len_0 = max_substring_length(s, numOps, '0')
        max_len_1 = max_substring_length(s, numOps, '1')
        
        return min(max_len_0, max_len_1)

# Example usage:
# sol = Solution()
# print(sol.minLength("000001", 1))  # Output: 2
# print(sol.minLength("0000", 2))    # Output: 1
# print(sol.minLength("0101", 0))    # Output: 1