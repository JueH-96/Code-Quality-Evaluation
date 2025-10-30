import collections

class Solution:
  def maxPalindromesAfterOperations(self, words: list[str]) -> int:
    # Calculate total character counts across all words
    char_counts = collections.Counter()
    for word in words:
      char_counts.update(word) # Efficiently counts characters in each word and aggregates
    
    # Calculate available pairs and single characters from the global pool.
    # These resources can be used to construct palindromes.
    available_pairs = 0
    available_singles = 0
    
    # Iterate through the counts of each unique character found
    for count in char_counts.values():
      available_pairs += count // 2  # Each pair of identical characters can form part of a palindrome
      available_singles += count % 2 # Any leftover characters are singles
      
    # Get lengths of all words. We want to make shorter words palindromic first
    # as they require fewer resources (pairs).
    word_lengths = []
    for word in words:
      word_lengths.append(len(word))
    word_lengths.sort() # Sort by length in non-decreasing order
    
    palindromes_count = 0
    
    # Iterate through sorted word lengths (shortest first)
    for length in word_lengths:
      # Calculate resources needed for a word of this 'length' to be a palindrome
      pairs_needed_for_symmetry = length // 2 # e.g., "abXba" needs 2 pairs (aa, bb) for "ab_ba" part
      
      if length % 2 == 1:  # Odd length word
        # Needs pairs_needed_for_symmetry and 1 single char for the center (e.g., 'X' in "abXba")
        
        # Attempt to use an existing single character for the center
        if available_singles >= 1:
          # If a single character is available for the center,
          # check if we have enough pairs for the symmetrical parts.
          if available_pairs >= pairs_needed_for_symmetry:
            # Resources are sufficient. Form the palindrome.
            available_pairs -= pairs_needed_for_symmetry
            available_singles -= 1
            palindromes_count += 1
          else:
            # Not enough pairs for symmetry. Cannot form this palindrome.
            # Since words are sorted by length, subsequent words will require at least as many pairs.
            # So, we cannot form any more palindromes.
            break 
        # If no existing single char, attempt to break a pair to create one for the center
        else:
          # To do this, we need pairs for symmetry AND one additional pair to break.
          # The broken pair (e.g., 'yy') yields two singles ('y', 'y').
          # One 'y' is used for the center. The other 'y' becomes an available_single.
          if available_pairs >= pairs_needed_for_symmetry + 1:
            # Resources are sufficient. Form the palindrome.
            available_pairs -= (pairs_needed_for_symmetry + 1) # Consumed pairs for symmetry + 1 for center
            available_singles += 1 # One character from broken pair becomes an available single
            palindromes_count += 1
          else:
            # Not enough pairs to do this. Cannot form this palindrome or any subsequent ones.
            break
            
      else:  # Even length word
        # Needs only pairs_needed_for_symmetry for the symmetrical parts (e.g., "abba" needs 2 pairs (aa, bb))
        if available_pairs >= pairs_needed_for_symmetry:
          # Resources are sufficient. Form the palindrome.
          available_pairs -= pairs_needed_for_symmetry
          palindromes_count += 1
        else:
          # Not enough pairs. Cannot form this palindrome or any subsequent ones.
          break
          
    return palindromes_count