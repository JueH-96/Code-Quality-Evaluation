class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        # Function to find the longest palindrome length from a single string
        def max_palindrome_length_from_string(st: str) -> int:
            char_count = {}
            for char in st:
                char_count[char] = char_count.get(char, 0) + 1
            
            length = 0
            odd_found = False
            
            for count in char_count.values():
                if count % 2 == 0:
                    length += count
                else:
                    length += count - 1
                    odd_found = True
            
            if odd_found:
                length += 1
            
            return length
        
        # Calculate the maximum palindrome length from both strings
        max_length_s = max_palindrome_length_from_string(s)
        max_length_t = max_palindrome_length_from_string(t)
        
        # The longest palindrome can be formed by combining the longest from s and t
        return max_length_s + max_length_t