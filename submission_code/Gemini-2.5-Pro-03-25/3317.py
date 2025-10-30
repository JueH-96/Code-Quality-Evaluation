import collections
from typing import List
import math # Not actually needed, integer division works fine.

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        """
        Solves the problem of finding the maximum number of palindromes achievable after operations.
        The swap operation allows characters to be rearranged freely across all strings. This means
        we can pool all characters together and redistribute them. The constraints are the total count 
        of each character ('a' through 'z') and the fixed lengths of the strings in `words`.
        A string can be rearranged into a palindrome if, among its characters, at most one character
        type appears an odd number of times.

        To maximize the number of palindromes, we should greedily try to form palindromes using the 
        shortest strings first, as they generally require fewer resources (character pairs).

        We calculate the total available resources: pairs of identical characters and leftover single 
        characters from the global character pool. Then, we iterate through the string lengths sorted 
        in ascending order, checking if we have enough resources to make the current string a palindrome.
        """
        
        # 1. Calculate total character counts across all words
        counts = collections.Counter()
        for word in words:
            # Add characters of the current word to the total counts
            counts.update(word)

        # 2. Compute total available pairs of identical characters and leftover single characters
        total_pairs = 0
        total_odd_chars = 0
        # Iterate through the counts of each character present in the input words
        for count in counts.values():
            # Each pair (c, c) contributes 1 to total_pairs
            total_pairs += count // 2
            # Each character with an odd count contributes 1 leftover single character
            total_odd_chars += count % 2

        # 3. Get the lengths of all words and sort them in ascending order
        # Processing shortest words first is a greedy strategy that ensures optimality.
        # Shorter strings require fewer pairs, which are often the limiting resource.
        lengths = [len(word) for word in words]
        lengths.sort()

        # 4. Iterate through sorted lengths greedily, trying to form palindromes
        palindrome_count = 0
        # Keep track of remaining resources available from the global pool
        rem_pairs = total_pairs
        rem_singles = total_odd_chars

        # Iterate through each length `l` in the sorted list of lengths
        for length in lengths:
            # Calculate resources needed to make a string of `length` into a palindrome
            # A palindrome of length `l` needs `l // 2` pairs of characters.
            pairs_needed = length // 2
            # If `l` is odd, it also needs one single character for the center.
            singles_needed = length % 2 # This is 1 if length is odd, 0 if even

            # Check if we have enough pairs available for the current word's palindrome structure
            if pairs_needed <= rem_pairs:
                # If enough pairs are available, tentatively reserve them.
                # Calculate the number of pairs remaining after fulfilling the paired character requirements.
                current_rem_pairs_after_pairing = rem_pairs - pairs_needed
                
                # Now check if we can satisfy the singles requirement (for the center of an odd length palindrome)
                if singles_needed == 1:
                    # A single character is needed for the center. Check if one is available.
                    if rem_singles >= 1:
                         # Yes, use one available single character. Commit resource usage.
                        rem_pairs = current_rem_pairs_after_pairing # Update remaining pairs
                        rem_singles -= 1 # Consume one single character
                        palindrome_count += 1 # Successfully formed a palindrome
                    else:
                        # No single characters available directly from the odd leftovers.
                        # We must try to obtain the needed single character by breaking a remaining pair.
                        # Breaking one pair yields two single characters. We need only one.
                        
                        # Check if there's at least one pair remaining that can be broken.
                        if current_rem_pairs_after_pairing >= 1:
                            # Yes, we can break a pair. Commit resource usage.
                            rem_pairs = current_rem_pairs_after_pairing - 1 # Break one pair
                            # Breaking a pair gives 2 singles. We use 1 for the center, so 1 single is left over.
                            # Net change to rem_singles: initially 0 available, +2 from broken pair, -1 used = +1
                            rem_singles += 1 
                            palindrome_count += 1 # Successfully formed a palindrome
                        else:
                            # Not enough singles initially, and no remaining pairs to break.
                            # We cannot satisfy the resource needs for this string. Stop the process.
                            break
                else: # singles_needed == 0 (the string has even length)
                    # No single character needed for the center. Commit the usage of pairs.
                    rem_pairs = current_rem_pairs_after_pairing # Update remaining pairs
                    # rem_singles remains unchanged as no single char was needed.
                    palindrome_count += 1 # Successfully formed a palindrome
            else:
                # Not enough pairs available even for the paired character requirements.
                # We cannot form a palindrome for this string or any subsequent longer strings. Stop.
                break
                
        # Return the total count of palindromes we could successfully form.
        return palindrome_count