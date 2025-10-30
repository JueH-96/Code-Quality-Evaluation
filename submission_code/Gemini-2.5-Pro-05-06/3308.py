import collections

class Solution:
  def lastNonEmptyString(self, s: str) -> str:
    n = len(s)
    
    # Constraints: 1 <= s.length <= 5 * 10^5
    # s consists only of lowercase English letters.
    # So, n is at least 1, s is not empty.

    # Step 1: Calculate frequencies of all characters in s.
    char_counts = collections.Counter(s)
    
    # Find the maximum frequency. Since s is non-empty, char_counts is non-empty
    # and char_counts.values() will provide a non-empty view of frequencies.
    max_freq = 0
    if char_counts: # This check is true due to constraints s.length >= 1
        max_freq = max(char_counts.values())
    # Edge case: if s was empty (not per constraints), max_freq would be 0.
    # An empty string implies 0 operations, and "string before last operation" is ill-defined.
    # However, with s.length >= 1, max_freq >= 1.
            
    # Step 2: (Implicit) Identify characters that will form the result.
    # These are characters `ch` such that their total frequency in `s` is `max_freq`.
    # We are interested in their `max_freq`-th occurrence in the original string `s`.
    
    # Step 3: Construct the result string.
    # Iterate through s from left to right.
    # Keep track of current occurrence count for each character encountered *during this scan*.
    # If a character `s[i]` is one of the characters that has overall frequency `max_freq`
    # AND this is its `max_freq`-th occurrence encountered in this scan,
    # then this character `s[i]` is part of the result.
    # These characters are collected in a list, maintaining their relative order from `s`.
    
    current_s_scan_counts = collections.Counter() # Stores counts for the current scan of s
    result_chars_list = []
    
    # Iterate through the string s from left to right
    for char_idx in range(n):
      char = s[char_idx]
      current_s_scan_counts[char] += 1
      
      # Check if this character's total frequency in s is max_freq
      # AND if this is its max_freq-th occurrence during this scan
      if char_counts[char] == max_freq and current_s_scan_counts[char] == max_freq:
        result_chars_list.append(char)
        
    return "".join(result_chars_list)