class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        def can_form_palindrome(word):
            # Count the frequency of each character in the word
            freq = {}
            for char in word:
                if char in freq:
                    freq[char] += 1
                else:
                    freq[char] = 1
            
            # Count how many characters have an odd frequency
            odd_count = sum(1 for count in freq.values() if count % 2 != 0)
            
            # A word can be rearranged into a palindrome if at most one character has an odd frequency
            return odd_count <= 1

        # Count how many words can be rearranged into palindromes
        palindrome_count = 0
        for word in words:
            if can_form_palindrome(word):
                palindrome_count += 1
        
        return palindrome_count