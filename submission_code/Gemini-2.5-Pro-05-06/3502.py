class Solution:
  def numberOfSubstrings(self, s: str, k: int) -> int:
    n = len(s)
    count = 0

    for i in range(n):
      # freq_map stores character counts for the current substring s[i:j+1].
      # It's reset for each new starting position i.
      freq_map = [0] * 26 
      
      for j in range(i, n):
        # Get the character s[j] and update its frequency
        char_idx = ord(s[j]) - ord('a')
        freq_map[char_idx] += 1
        
        # Check if the character s[j] (whose count was just incremented)
        # now has a count of at least k.
        if freq_map[char_idx] >= k:
          # If this condition is met, it means s[i:j+1] is a valid substring.
          # Since we iterate j from i upwards, this is the first (shortest)
          # substring starting at i that satisfies the condition.
          
          # Any substring s[i:p+1] where p >= j will also be valid.
          # This is because the character s[j] (or more accurately, the character
          # type of s[j]) will maintain its frequency of at least k
          # in these longer substrings starting at i.
          
          # The number of such substrings is (n - 1) - j + 1 = n - j.
          # These are s[i:j+1], s[i:j+2], ..., s[i:n].
          count += (n - j)
          
          # We have found the smallest j for this i that makes s[i:j+1] valid
          # and have added all relevant counts (n-j) for this i.
          # So, we can break from the inner loop and proceed to the next starting position i+1.
          break
          
    return count