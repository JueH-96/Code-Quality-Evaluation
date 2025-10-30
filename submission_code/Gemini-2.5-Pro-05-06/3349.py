import collections

class Solution:
  def maximumLengthSubstring(self, s: str) -> int:
    n = len(s)
    max_length = 0

    # Iterate over all possible starting positions i for a substring
    for i in range(n):
      # For each starting position i, counts will store frequencies
      # of characters in the substring s[i...j].
      # Initialize counts for characters 'a' through 'z'.
      counts = [0] * 26 # Reset counts for each new starting position i
      
      # Iterate over all possible ending positions j for a substring starting at i
      for j in range(i, n):
        # Current character being added to the substring s[i...j]
        char = s[j]
        # Convert character to a 0-25 index (e.g., 'a' -> 0, 'b' -> 1, ...)
        char_idx = ord(char) - ord('a') 
        
        counts[char_idx] += 1
        
        # Check if the count of the current character exceeds 2
        if counts[char_idx] > 2:
          # If s[j] causes its count to be > 2, then substring s[i...j] is invalid.
          # Any further extension s[i...k] (where k > j) would also be invalid
          # because it would still contain s[j] (and thus its character type) three or more times.
          # So, we break from this inner loop (for j) and move to the next starting position i+1.
          break
        else:
          # If counts[char_idx] <= 2, this character s[j] does not violate the condition.
          # The substring s[i...j-1] (if j > i) was guaranteed to be valid by previous iterations 
          # of this inner loop (its character counts were all <= 2).
          # Since only counts[char_idx] changed, and it's still <= 2,
          # all character counts in s[i...j] must be <= 2. So, s[i...j] is valid.
          current_length = j - i + 1
          if current_length > max_length:
            max_length = current_length
            
    return max_length