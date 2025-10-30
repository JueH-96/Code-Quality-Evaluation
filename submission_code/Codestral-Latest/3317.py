class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        def is_palindrome(s: str) -> bool:
            return s == s[::-1]

        # Count the number of palindromes in the initial list
        initial_palindromes = sum(1 for word in words if is_palindrome(word))

        # Since we can perform any number of swaps, we can always make all words palindromes
        # by swapping characters within each word to form a palindrome.
        return len(words)