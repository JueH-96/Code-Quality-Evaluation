class Solution:
  def hasMatch(self, s: str, p: str) -> bool:
    star_idx = p.find('*')
    # p is guaranteed to have exactly one '*'
    
    prefix_p = p[:star_idx]
    suffix_p = p[star_idx+1:]
    
    len_s = len(s)
    # Store lengths for clarity and potential minor optimization if len() is expensive.
    len_prefix_p = len(prefix_p)
    len_suffix_p = len(suffix_p)
    
    # Iterate through all possible starting positions of the pattern match in s.
    # `i` is the starting index in `s` where `prefix_p` (and thus the whole pattern) could match.
    # `i` can go up to `len_s` because `prefix_p` could be empty, 
    # and `s.startswith("", len_s)` is true.
    for i in range(len_s + 1):
      # Check if prefix_p matches s[i : i + len_prefix_p]
      # s.startswith(prefix, index_in_s) checks if s[index_in_s:] starts with prefix.
      # This correctly handles bounds: e.g. if i + len_prefix_p > len_s, 
      # s.startswith will be false (unless prefix_p is empty).
      if not s.startswith(prefix_p, i):
        continue
      
      # If prefix_p matches at s[i...], then this part of s is s[i : i + len_prefix_p].
      # The characters for '*' would start matching from index (i + len_prefix_p) in s.
      # This is also the earliest possible start for suffix_p if '*' matches an empty string.
      start_index_for_middle_or_suffix = i + len_prefix_p
      
      # Now, look for suffix_p in s. 
      # suffix_p must start at or after start_index_for_middle_or_suffix.
      # (The characters between the end of prefix_p and start of suffix_p form the match for '*')
      # `j` is the starting index in `s` where `suffix_p` could match.
      # `j` can go up to `len_s` because `suffix_p` could be empty.
      for j in range(start_index_for_middle_or_suffix, len_s + 1):
        # Check if suffix_p matches s[j : j + len_suffix_p]
        if not s.startswith(suffix_p, j):
          continue
          
        # If suffix_p matches at s[j...], then this part of s is s[j : j + len_suffix_p].
        # The end of the matched substring in s (exclusive) is (j + len_suffix_p).
        end_index_of_matched_substring = j + len_suffix_p
        
        # The full substring in s that could match p is s[i : end_index_of_matched_substring].
        # Its length is (end_index_of_matched_substring - i).
        
        # According to the problem: "A substring is a contiguous non-empty sequence of characters".
        # So, the matched part of s must be non-empty.
        current_substring_length = end_index_of_matched_substring - i
        
        if current_substring_length > 0:
          # Found a non-empty substring of s that matches the pattern p.
          # The pattern p, when '*' is appropriately replaced, becomes equal to
          # the substring s[i : end_index_of_matched_substring].
          # This substring s[i : end_index_of_matched_substring] is guaranteed to be non-empty by this check.
          return True
          
    # If no such match is found after checking all possibilities
    return False