from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count frequency of each character
        count = [0] * 26
        for word in words:
            for char in word:
                count[ord(char) - ord('a')] += 1
        
        # Calculate total pairs and singles
        total_pairs = 0
        total_singles = 0
        for freq in count:
            total_pairs += freq // 2
            total_singles += freq % 2
        
        # Sort words by length
        words.sort(key=len)
        
        # Initialize palindrome count
        palindrome_count = 0
        
        # Iterate through each word
        for word in words:
            length = len(word)
            if length % 2 == 0:
                # Even length: needs length//2 pairs
                needed_pairs = length // 2
                if total_pairs >= needed_pairs:
                    total_pairs -= needed_pairs
                    palindrome_count += 1
            else:
                # Odd length: needs length//2 pairs and 1 single
                needed_pairs = length // 2
                needed_singles = 1
                if total_pairs >= needed_pairs and total_singles >= needed_singles:
                    total_pairs -= needed_pairs
                    total_singles -= needed_singles
                    palindrome_count += 1
                elif needed_pairs == 0 and needed_singles == 1 and total_singles >= 1:
                    # Special case: single character word
                    total_singles -= 1
                    palindrome_count += 1
        
        return palindrome_count