import collections
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Step 1: Count character frequencies for all characters across all words.
        char_counts = collections.Counter()
        for word in words:
            char_counts.update(word)

        # Step 2: Calculate total available character pairs and single characters.
        # current_pairs: total number of character pairs (e.g., 'aa', 'bb') available.
        # current_singles: total number of single, unpaired characters available.
        current_pairs = 0
        current_singles = 0
        for count in char_counts.values():
            current_pairs += count // 2  # Each pair uses two characters
            current_singles += count % 2 # Remaining single characters
        
        # Step 3: Get lengths of all words and sort them in ascending order.
        # This greedy approach prioritizes satisfying shorter strings first,
        # as they require fewer pairs, thus maximizing the chance to form more palindromes.
        lengths = sorted([len(word) for word in words])
        
        palindromes_count = 0
        
        # Step 4: Iterate through sorted lengths and try to form palindromes.
        for L in lengths:
            pairs_for_this_string = L // 2
            single_for_this_string_needed = L % 2 # 1 if L is odd, 0 if L is even
            
            # Check if we have enough pairs to form the symmetric sides of the palindrome.
            if current_pairs >= pairs_for_this_string:
                current_pairs -= pairs_for_this_string
                
                # If the string length is odd, it also needs a single character for its middle.
                if single_for_this_string_needed == 1:
                    # Prefer to use an existing single character if available.
                    if current_singles > 0:
                        current_singles -= 1
                        palindromes_count += 1
                    # If no existing single characters are available, try to break one of our pairs.
                    # Breaking a pair provides two single characters; one is used for the middle
                    # of the current palindrome, and the other effectively replaces the broken pair
                    # as a new single character (meaning `current_singles` remains conceptually unchanged at 0
                    # if it was already 0, as one is used and one is gained).
                    elif current_pairs > 0:
                        current_pairs -= 1
                        palindromes_count += 1
                    else:
                        # Cannot form this palindrome:
                        # We had enough pairs for the sides, but no single characters (current_singles == 0)
                        # and no pairs left to break (current_pairs == 0) for the middle character.
                        # Thus, this palindrome cannot be formed. We must return the pairs
                        # that were optimistically consumed for its sides back to the pool.
                        current_pairs += pairs_for_this_string
                        # Do not increment palindromes_count.
                else:
                    # Even length string: no middle character is needed.
                    # Palindrome can be successfully formed.
                    palindromes_count += 1
            else:
                # Not enough pairs to even form the sides of this palindrome.
                # Cannot form this palindrome.
                # No resources were consumed, so no need to adjust counts.
                pass 
        
        # Step 5: Return the maximum number of palindromes achievable.
        return palindromes_count