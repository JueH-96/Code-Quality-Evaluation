import collections

class Solution:
  def maxDifference(self, s: str) -> int:
    """
    Calculates the maximum difference between the frequency of two characters 
    in a string, one with odd frequency and one with even frequency.
    The difference is (frequency of odd-frequency character) - (frequency of even-frequency character).
    """
    
    # 1. Calculate character frequencies.
    # collections.Counter efficiently creates a frequency map.
    # For example, s = "aaaaabbc" -> counts = {'a': 5, 'b': 2, 'c': 1}
    counts = collections.Counter(s)
    
    # 2. Initialize variables to find the maximum odd frequency 
    #    and minimum non-zero even frequency.
    # Initialize max_odd_freq to negative infinity, as any odd frequency (>=1) will be greater.
    max_odd_freq = float('-inf')
    # Initialize min_even_freq to positive infinity, as any even frequency (>=2) will be smaller.
    min_even_freq = float('inf')
    
    # 3. Iterate through the frequencies of characters present in the string.
    # counts.values() returns an iterable of the frequencies (e.g., [5, 2, 1] for "aaaaabbc").
    # The order of frequencies does not affect finding the overall max odd and min even frequency.
    for freq in counts.values():
      # Frequencies from counts.values() are for characters present in s, so freq > 0.
      if freq % 2 == 1:  # If the frequency is odd
        # Update max_odd_freq if the current odd frequency is larger.
        if freq > max_odd_freq:
          max_odd_freq = freq
      else:  # If the frequency is even (freq % 2 == 0)
        # Update min_even_freq if the current even frequency is smaller.
        if freq < min_even_freq:
          min_even_freq = freq
          
    # 4. Calculate and return the maximum difference.
    # The problem constraints state:
    # "s contains at least one character with an odd frequency 
    #  and one with an even frequency."
    # This guarantees that max_odd_freq will have been updated to an actual odd frequency value
    # from the string, and min_even_freq will have been updated to an actual even frequency value.
    # Thus, they will not remain at their initial float('-inf') or float('inf') values.
    
    return max_odd_freq - min_even_freq