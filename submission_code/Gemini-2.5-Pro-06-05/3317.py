import collections
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        """
        Calculates the maximum number of palindromic strings that can be formed
        by swapping characters between any of the strings in the input list.
        """
        
        # Step 1: Count all characters across all words to create a global pool.
        char_counts = collections.Counter("".join(words))
        
        # Step 2: Calculate the total available pairs and single characters.
        # A character with count 'c' contributes c // 2 pairs and c % 2 singles.
        total_pairs = 0
        total_singles = 0
        for count in char_counts.values():
            total_pairs += count // 2
            total_singles += count % 2
            
        # Step 3: Sort word lengths to process shorter (cheaper) words first.
        lengths = sorted(len(w) for w in words)
        
        # Step 4: Greedily try to form palindromes.
        palindromes_count = 0
        for length in lengths:
            # Calculate resources needed for a palindrome of this length.
            pairs_needed = length // 2
            singles_needed = length % 2
            
            # First, check if we have enough pairs.
            if total_pairs < pairs_needed:
                # Not enough pairs. Since lengths are sorted, we can't form
                # this or any subsequent (longer) palindromes.
                break
            
            # Tentatively use the pairs for the current word.
            total_pairs -= pairs_needed
            
            # Now, check if we can satisfy the single character requirement for odd lengths.
            if singles_needed == 1:
                if total_singles > 0:
                    # Use an available single character.
                    total_singles -= 1
                elif total_pairs > 0:
                    # No singles available, so we break a pair.
                    # This costs 1 pair and gives 2 singles. We use one for the center,
                    # and the other is added back to our pool. Net effect: -1 pair, +1 single.
                    total_pairs -= 1
                    total_singles += 1
                else:
                    # Cannot satisfy the need for a single character. We must stop.
                    # We need to revert the pairs we took for this attempt.
                    total_pairs += pairs_needed
                    break
            
            # If we successfully allocated all resources for this length, it's a palindrome.
            palindromes_count += 1
            
        return palindromes_count