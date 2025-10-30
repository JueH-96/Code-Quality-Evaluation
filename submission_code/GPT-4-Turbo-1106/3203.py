from typing import List
from collections import Counter

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        # Helper function to get the frequency count of characters in a substring
        def get_freq(s, start, end):
            return Counter(s[start:end+1])
        
        # Helper function to check if a string can be rearranged to form a palindrome
        def can_form_palindrome(freq):
            odd_count = sum(1 for count in freq.values() if count % 2 != 0)
            return odd_count <= 1
        
        # Main function logic
        n = len(s)
        answer = []
        for a, b, c, d in queries:
            # Get the frequency count of characters in both substrings
            freq_left = get_freq(s, a, b)
            freq_right = get_freq(s, c, d)
            
            # Combine the frequencies to get the total frequency count
            total_freq = freq_left + freq_right
            
            # Check if the combined string can be rearranged to form a palindrome
            answer.append(can_form_palindrome(total_freq))
        
        return answer

# Example usage:
# sol = Solution()
# print(sol.canMakePalindromeQueries("abcabc", [[1,1,3,5],[0,2,5,5]]))  # Output: [True, True]
# print(sol.canMakePalindromeQueries("abbcdecbba", [[0,2,7,9]]))       # Output: [False]
# print(sol.canMakePalindromeQueries("acbcab", [[1,2,4,5]]))           # Output: [True]