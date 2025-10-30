from typing import List

class Solution:
  def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
    forbidden_set = set(forbidden)
    
    current_max_f_len = 0
    # Constraints: 1 <= forbidden.length, 1 <= forbidden[i].length.
    # So forbidden_set is non-empty and strings in it have length >= 1.
    # Thus, current_max_f_len will be >= 1 after this loop.
    if not forbidden_set: # Should not be reached given constraints but good for robustness
        return len(word)
        
    for f_str in forbidden_set: 
        current_max_f_len = max(current_max_f_len, len(f_str))
    
    n = len(word)
    max_valid_len = 0
    
    # left_pointer stores the smallest index L such that word[L : right+1] is a valid substring.
    # It's effectively 1 + (the largest end index of a forbidden substring encountered so far,
    # that affects the current window).
    # More simply, it's max(s_i + 1) for all forbidden word[s_i:e_i+1] where e_i <= right.
    left_pointer = 0 
    
    for right in range(n):
      # Check for forbidden substrings ending at `right`.
      # Iterate k from `right` down towards `0`.
      # `k` is the potential start index of a substring `word[k : right+1]`.
      for k in range(right, -1, -1):
        # Condition 1: Substring length check
        # If `word[k : right+1]` is longer than any forbidden string,
        # it (and any substrings starting earlier, i.e. smaller k) cannot be in `forbidden_set`.
        if (right - k + 1) > current_max_f_len:
          break 
        
        # Condition 2: `left_pointer` check
        # If k < left_pointer, then this k is already to the left of the
        # earliest allowed start for a valid window.
        # The update `max(left_pointer, k + 1)` would be `left_pointer` itself,
        # as `k + 1 <= left_pointer`.
        # Since k is decreasing, further smaller k values also won't advance `left_pointer`.
        if k < left_pointer:
          break

        # substring_to_check is word[k : right+1]
        # This slice is at most current_max_f_len characters long.
        substring_to_check = word[k : right+1] 
        if substring_to_check in forbidden_set:
          # If word[k : right+1] is forbidden, then any valid substring
          # ending at `right` must start after `k`. So, its L must be at least `k+1`.
          left_pointer = max(left_pointer, k + 1)
          
      # After checking all relevant substrings ending at `right`,
      # `left_pointer` is the smallest index L such that `word[L : right+1]` is valid.
      # The length of this valid substring is `right - left_pointer + 1`.
      current_length = right - left_pointer + 1
      max_valid_len = max(max_valid_len, current_length)
      
    return max_valid_len