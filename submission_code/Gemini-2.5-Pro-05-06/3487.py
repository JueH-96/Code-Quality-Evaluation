class Solution:
  def maxRemovals(self, source: str, pattern: str, targetIndices: list[int]) -> int:
    n_s = len(source)
    n_p = len(pattern)

    # This boolean array will keep track of which characters are removed.
    # Initially, all False (no characters removed).
    # Indices in source string correspond to indices in this array.
    removed_mask = [False] * n_s

    # Helper function to check if pattern is a subsequence of source
    # considering the characters marked in removed_mask.
    def is_subsequence_check() -> bool:
      s_ptr, p_ptr = 0, 0 # Pointers for source and pattern respectively
      while s_ptr < n_s and p_ptr < n_p:
        # If current source character is marked as removed, skip it.
        if removed_mask[s_ptr]:
          s_ptr += 1
          continue
        
        # If current source character (not removed) matches current pattern character
        if source[s_ptr] == pattern[p_ptr]:
          p_ptr += 1 # Move to next character in pattern
        s_ptr += 1 # Always move to next character in source
      
      # If p_ptr reached end of pattern, all characters of pattern were found in order
      return p_ptr == n_p

    num_ops = 0
    # Iterate through the targetIndices. For each index, try to remove the character at that index.
    # The problem states targetIndices is sorted. We process them in this given order.
    for i in range(len(targetIndices)):
      idx_to_try_removing = targetIndices[i]
      
      # Tentatively mark this character as removed
      removed_mask[idx_to_try_removing] = True
      
      # Check if pattern is still a subsequence after this removal
      if is_subsequence_check():
        # If yes, this removal is valid according to the problem's definition of an operation.
        # Increment operation count. The character remains marked as removed for subsequent checks.
        num_ops += 1 
      else:
        # If no, this removal invalidates the subsequence property.
        # So, we cannot perform this operation. Revert the change (mark as not removed).
        removed_mask[idx_to_try_removing] = False
        
    return num_ops