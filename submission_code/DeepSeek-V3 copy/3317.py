from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count the frequency of each character across all words
        total_counts = Counter()
        for word in words:
            total_counts.update(word)
        
        # Calculate the total number of characters
        total_chars = sum(total_counts.values())
        
        # Calculate the number of characters that can be used to form pairs
        # Each pair contributes to the even part of the palindrome
        pair_chars = 0
        for char, cnt in total_counts.items():
            pair_chars += cnt // 2
        
        # Calculate the maximum number of palindromes
        # Each palindrome requires at least one pair of characters (for even length) or one pair plus one single character (for odd length)
        # The number of palindromes is limited by the number of pairs and the number of words
        max_palindromes = 0
        for word in words:
            length = len(word)
            # Calculate the number of pairs needed for this word
            pairs_needed = length // 2
            if pairs_needed <= pair_chars:
                max_palindromes += 1
                pair_chars -= pairs_needed
            else:
                break
        
        return max_palindromes