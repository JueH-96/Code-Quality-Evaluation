from typing import List
from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count total characters across all words
        total_counts = Counter(''.join(words))
        
        # Calculate total available pairs and single characters
        total_pairs = sum(count // 2 for count in total_counts.values())
        total_singles = sum(count % 2 for count in total_counts.values())
        
        # Prepare list of word requirements: (pairs_needed, single_needed)
        word_requirements = []
        for word in words:
            length = len(word)
            pairs_needed = length // 2
            single_needed = length % 2
            word_requirements.append((pairs_needed, single_needed))
        
        # Sort words by pairs_needed ascending, then single_needed ascending
        word_requirements.sort(key=lambda x: (x[0], x[1]))
        
        assigned_palindromes = 0
        available_pairs = total_pairs
        available_singles = total_singles
        
        for pairs_needed, single_needed in word_requirements:
            if available_pairs >= pairs_needed:
                available_pairs -= pairs_needed
                if single_needed:
                    if available_singles >= 1:
                        available_singles -= 1
                    else:
                        if available_pairs >= 1:
                            available_pairs -= 1
                            available_singles += 1  # Convert a pair into two singles
                            available_singles -= 1
                        else:
                            # Not enough singles or pairs to assign single character
                            available_pairs += pairs_needed
                            continue
                assigned_palindromes += 1
        
        return assigned_palindromes