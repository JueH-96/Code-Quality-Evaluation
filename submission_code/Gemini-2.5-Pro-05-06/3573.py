class Solution:
  def validSubstringCount(self, word1: str, word2: str) -> int:
    n = len(word1)
    m = len(word2)

    if m == 0:
      # If word2 is empty, any non-empty string x can be rearranged 
      # to have word2 (empty string) as a prefix.
      # So, all non-empty substrings of word1 are valid.
      # The number of non-empty substrings of a string of length n is n * (n + 1) // 2.
      return n * (n + 1) // 2
    
    target_counts = [0] * 26
    for char_w2 in word2:
      target_counts[ord(char_w2) - ord('a')] += 1

    current_counts = [0] * 26
    
    # diff_count represents the total number of characters "missing" from the current window
    # to satisfy the target_counts.
    # A window is valid if diff_count is 0.
    # Initially, current_counts is all zeros. So, diff_count is sum of target_counts[c], which is m.
    diff_count = m 
    
    ans = 0
    # right_ptr is the exclusive end of the current window [left_ptr, right_ptr-1]
    right_ptr = 0 

    for left_ptr in range(n):
      # Expand window by moving right_ptr.
      # The window being built/checked is word1[left_ptr ... right_ptr-1].
      # We try to add word1[right_ptr] to this window.
      while right_ptr < n and diff_count > 0:
        char_idx_added = ord(word1[right_ptr]) - ord('a')
        
        # If current_counts for this character is less than its target_counts,
        # then adding this character helps satisfy its part of the target.
        if current_counts[char_idx_added] < target_counts[char_idx_added]:
          diff_count -= 1
        current_counts[char_idx_added] += 1
        
        right_ptr += 1
      
      # After the expansion loop, the window is word1[left_ptr ... right_ptr-1].
      # The loop terminated if:
      # 1. diff_count became 0: The window word1[left_ptr ... right_ptr-1] is valid.
      #    This is the shortest valid window starting at left_ptr.
      # 2. right_ptr reached n: We exhausted word1. diff_count might still be > 0.

      if diff_count == 0:
        # The window word1[left_ptr ... right_ptr-1] is the shortest valid one starting at left_ptr.
        # Its end index is right_ptr-1.
        # Any superstring starting at left_ptr, i.e., word1[left_ptr ... k] where k >= (right_ptr-1),
        # will also be valid (as it contains all characters of the current valid window plus more).
        # These valid ending indices k range from (right_ptr-1) to (n-1).
        # The number of such valid substrings is (n-1) - (right_ptr-1) + 1 = n - right_ptr + 1.
        ans += (n - right_ptr + 1)
      else: # diff_count > 0
        # This implies right_ptr == n (because if right_ptr < n and diff_count > 0,
        # the while loop for expansion would have continued).
        # Since we've exhausted word1 (right_ptr == n) and still diff_count > 0,
        # no valid window starting at left_ptr could be formed.
        # Furthermore, for any subsequent left_ptr' ( > left_ptr ), right_ptr cannot advance further.
        # Removing characters from the left will only increase diff_count or keep it similar.
        # So, no more valid substrings can be found. We can break the main loop.
        break 
            
      # Shrink window from the left: effectively remove word1[left_ptr].
      # This prepares for the next iteration of the main loop (next left_ptr).
      char_idx_removed = ord(word1[left_ptr]) - ord('a')
      
      current_counts[char_idx_removed] -= 1
      # If, after decrementing, current_counts for this character falls below its target_counts,
      # then a "deficit" for this character type is created or increased.
      if current_counts[char_idx_removed] < target_counts[char_idx_removed]:
        diff_count += 1
        
    return ans