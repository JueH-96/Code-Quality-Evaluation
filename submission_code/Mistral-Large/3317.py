from collections import Counter
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        def can_form_palindrome(word: str) -> bool:
            # A word can form a palindrome if it has at most one character with an odd frequency
            freq = Counter(word)
            odd_count = sum(1 for count in freq.values() if count % 2 != 0)
            return odd_count <= 1

        palindrome_count = 0
        non_palindromes = []

        # Count already palindromic words
        for word in words:
            if word == word[::-1]:
                palindrome_count += 1
            else:
                non_palindromes.append(word)

        # Check if non-palindromic words can be rearranged to form palindromes
        for word in non_palindromes:
            if can_form_palindrome(word):
                palindrome_count += 1

        return palindrome_count

# Example usage:
# solution = Solution()
# print(solution.maxPalindromesAfterOperations(["abbb","ba","aa"]))  # Output: 3
# print(solution.maxPalindromesAfterOperations(["abc","ab"]))        # Output: 2
# print(solution.maxPalindromesAfterOperations(["cd","ef","a"]))      # Output: 1