class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        def is_palindrome(s):
            return s == s[::-1]

        n = len(words)
        palindromes = 0
        single_chars = 0
        pairs = {}

        for word in words:
            if is_palindrome(word):
                palindromes += 1
            else:
                # Count character frequencies
                for char in word:
                    if char in pairs:
                        pairs[char] += 1
                    else:
                        pairs[char] = 1

        # Count pairs of characters that can form palindromes
        for count in pairs.values():
            palindromes += count // 2
            single_chars += count % 2

        # If we have any single characters left, we can form one more palindrome
        if single_chars > 0:
            palindromes += 1

        return palindromes