class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        def can_form_palindrome(s):
            count = [0] * 26
            for char in s:
                count[ord(char) - ord('a')] += 1
            odd_count = sum(1 for c in count if c % 2 != 0)
            return odd_count <= 1

        palindrome_count = 0
        for word in words:
            if can_form_palindrome(word):
                palindrome_count += 1
            else:
                freq = [0] * 26
                for char in word:
                    freq[ord(char) - ord('a')] += 1
                odd_chars = [i for i in range(26) if freq[i] % 2 != 0]
                if len(odd_chars) <= 1:
                    palindrome_count += 1
                else:
                    if len(odd_chars) % 2 == 0:
                        palindrome_count += 1

        return palindrome_count