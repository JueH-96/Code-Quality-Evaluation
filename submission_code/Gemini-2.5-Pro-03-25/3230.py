import math # Not actually needed, but was in starter code exploration. Can be removed.

class Solution:
  def removeAlmostEqualCharacters(self, word: str) -> int:
      """
      Calculates the minimum number of operations to make sure no adjacent characters
      in the word are almost-equal using a greedy approach.

      Two characters are almost-equal if they are the same or adjacent in the alphabet.
      An operation consists of changing a character at any index to any lowercase English letter.

      The greedy strategy iterates through the string and whenever an almost-equal pair
      is found at indices (i, i+1), it increments the operation count and skips checking
      the next pair (i+1, i+2) by advancing the index i by 2. If the pair is not almost-equal,
      it advances the index i by 1.
      """

      # Helper function defined inside the method for encapsulation and clarity.
      def are_almost_equal(char1: str, char2: str) -> bool:
          """
          Checks if two characters char1 and char2 are almost-equal.
          Almost-equal means they are the same character or adjacent in the alphabet.
          This condition is equivalent to the absolute difference of their ASCII values being <= 1.
          """
          # ord() returns the ASCII value of a character.
          # For lowercase English letters ('a' through 'z'), their ASCII values are consecutive integers.
          return abs(ord(char1) - ord(char2)) <= 1

      count = 0 # Initialize the count of operations needed.
      i = 0 # Initialize the pointer to the current index being checked.
      n = len(word) # Cache the length of the word for efficiency.
      
      # Iterate through the string using the pointer i.
      # The loop continues as long as i can represent the first index of a valid pair (i, i+1).
      # Thus, the maximum value for i is n-2. The loop condition is i < n - 1.
      while i < n - 1:
          # Check if the character at index i and the character at index i+1 are almost-equal.
          if are_almost_equal(word[i], word[i+1]):
              # If they are almost-equal, we must perform an operation to change one of them.
              # The greedy choice is to count one operation (conceptually, we change word[i+1]).
              count += 1
              
              # Since we performed an operation involving index i+1, we have resolved the conflict
              # between indices i and i+1. To avoid potential issues with the character we just
              # conceptually changed at index i+1, and to ensure each operation addresses disjoint issues,
              # we skip checking the pair (word[i+1], word[i+2]).
              # The next pair we need to check starts at index i+2.
              # So, we advance the pointer i by 2 positions.
              i += 2 
          else:
              # If the characters at indices i and i+1 are not almost-equal, this pair is already valid.
              # We don't need to perform any operation for this pair.
              # Move to check the next pair, which starts at index i+1.
              # Advance the pointer i by 1 position.
              i += 1
              
      # After iterating through the string, 'count' holds the total minimum number of operations required.
      return count