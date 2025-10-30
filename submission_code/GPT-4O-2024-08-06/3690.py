class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        def max_length_with_flips(s, numOps, target_char):
            max_len = 0
            left = 0
            flips = 0
            
            for right in range(len(s)):
                if s[right] != target_char:
                    flips += 1
                
                while flips > numOps:
                    if s[left] != target_char:
                        flips -= 1
                    left += 1
                
                max_len = max(max_len, right - left + 1)
            
            return max_len
        
        max_zeros = max_length_with_flips(s, numOps, '0')
        max_ones = max_length_with_flips(s, numOps, '1')
        
        return min(max_zeros, max_ones)

# Example usage:
# solution = Solution()
# print(solution.minLength("000001", 1))  # Output: 2
# print(solution.minLength("0000", 2))    # Output: 1
# print(solution.minLength("0101", 0))    # Output: 1