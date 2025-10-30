import math # Used for float('inf')

class Solution:
  """
  This class provides a solution to find the maximum difference between
  the frequency of an odd-occurring character and an even-occurring character
  in a given string, conforming to the problem specification.
  """
  def maxDifference(self, s: str) -> int:
    """
    Calculates the maximum difference between the frequency of a character
    with odd frequency and a character with even frequency in the string s.

    The difference is defined as freq(odd_char) - freq(even_char). The goal is
    to find the maximum possible value of this difference over all pairs of
    characters where one has an odd frequency and the other has an even frequency.

    Args:
      s: The input string consisting of lowercase English letters.
         Constraints:
           - 3 <= len(s) <= 100
           - s consists only of lowercase English letters.
           - s is guaranteed to contain at least one character with an odd
             frequency and one with an even frequency.

    Returns:
      The maximum difference, calculated as the frequency of the character
      with the highest odd frequency minus the frequency of the character
      with the lowest even frequency.
    """

    # Use a fixed-size array to store frequencies of the 26 lowercase English letters.
    # Initialize all frequencies to 0. The index corresponds to the character:
    # index 0 for 'a', 1 for 'b', ..., 25 for 'z'.
    freq = [0] * 26

    # Calculate the frequency of each character in the input string s.
    # This loop runs len(s) times.
    for char in s:
      # Determine the array index for the current character.
      index = ord(char) - ord('a')
      # Increment the frequency count for this character.
      freq[index] += 1

    # Initialize variables to store the maximum odd frequency and minimum even frequency encountered.
    
    # Initialize max_odd_freq to 0. Since frequencies are positive, and the smallest
    # possible odd frequency is 1, this serves as a safe lower bound. Any actual odd
    # frequency found will be >= 1 and will correctly update this variable.
    max_odd_freq = 0
    
    # Initialize min_even_freq to positive infinity. This ensures that the first
    # encountered even frequency (which must be >= 2) will become the initial minimum.
    # Using float('inf') avoids needing to import the math module.
    min_even_freq = float('inf') 

    # Iterate through the frequency counts for each of the 26 possible characters.
    # This loop runs a constant 26 times.
    for count in freq:
      # We only process characters that actually appear in the string (i.e., have frequency > 0).
      if count > 0:
        # Check if the frequency count is even.
        if count % 2 == 0:
          # If the frequency is even, we potentially update the minimum even frequency found so far.
          min_even_freq = min(min_even_freq, count)
        # If the frequency count is not even, it must be odd.
        else:
          # If the frequency is odd, we potentially update the maximum odd frequency found so far.
          max_odd_freq = max(max_odd_freq, count)

    # The problem constraints guarantee that the string 's' contains at least one character
    # with an odd frequency and at least one character with an even frequency.
    # This ensures that after the loop:
    # - max_odd_freq will hold the maximum frequency among all characters that appear an odd number of times (value >= 1).
    # - min_even_freq will hold the minimum frequency among all characters that appear an even number of times (value >= 2 and finite).

    # Calculate the maximum difference as specified by the problem:
    # (maximum odd frequency) - (minimum even frequency).
    result = max_odd_freq - min_even_freq

    # Return the calculated maximum difference.
    return result