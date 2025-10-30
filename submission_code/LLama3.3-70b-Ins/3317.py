from collections import Counter
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count the frequency of each character across all words
        char_count = Counter(''.join(words))
        
        # Initialize the count of palindromes and the count of odd frequency characters
        palindrome_count = 0
        odd_count = 0
        
        # Iterate over the character counts
        for count in char_count.values():
            # If the count is odd, increment the odd count
            if count % 2 != 0:
                odd_count += 1
        
        # The maximum number of palindromes is the length of the words list minus the number of odd counts divided by 2
        # This is because each pair of odd counts can be combined to form a palindrome
        palindrome_count = len(words) - (odd_count + 1) // 2
        
        return palindrome_count