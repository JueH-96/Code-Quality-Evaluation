class Solution:
  def _count_partitions_core(self, text_list: list, k_val: int, n_text: int) -> int:
      if n_text == 0:
          return 0
      
      num_partitions = 0
      # current_idx marks the start of the text segment for which we are currently finding a partition.
      current_idx = 0 
      
      # Reusable frequency array for characters in the current partition attempt.
      # Max alphabet size is 26 for lowercase English letters.
      freq = [0] * 26 
      
      while current_idx < n_text:
          num_partitions += 1
          
          # Clear frequency counts for the new partition.
          # This can be done by iterating 0..25 and setting freq[j]=0,
          # or by re-initializing freq = [0]*26. The latter might be slightly less efficient
          # due to memory allocation, but conceptually simple. Iterating is fine.
          for i_char_code in range(26):
              freq[i_char_code] = 0
          
          distinct_count = 0
          
          # p_ptr scans from current_idx to find the end of the current partition.
          p_ptr = current_idx 
          while p_ptr < n_text:
              char_ord_val = ord(text_list[p_ptr]) - ord('a') # Get 0-25 code for char
              
              if freq[char_ord_val] == 0: # If this char is new to the current partition
                  distinct_count += 1
              freq[char_ord_val] += 1 # Increment its frequency for this partition
              
              if distinct_count > k_val:
                  # The current character text_list[p_ptr] makes distinct_count exceed k_val.
                  # So, the current partition is text_list[current_idx ... p_ptr-1].
                  # The next partition search will start from p_ptr.
                  current_idx = p_ptr 
                  break # Exit inner while loop (p_ptr loop)
              else:
                  # This character can be included. Continue extending the current partition.
                  p_ptr += 1
          
          # If the inner loop completed because p_ptr reached n_text:
          # This means the segment text_list[current_idx ... n_text-1] forms the current partition,
          # as it did not exceed k_val distinct characters.
          if p_ptr == n_text: # Check if loop terminated by reaching end of text
              current_idx = n_text # All of text_list processed.
              # The outer while loop will terminate as current_idx is now n_text.
      
      return num_partitions

  def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
      n = len(s)
      # Constraints: 1 <= s.length <= 10^4. So n won't be 0.
      # if n == 0: return 0 # Defensive check, not strictly needed by constraints.
          
      s_list = list(s) # Work with a list of characters for O(1) item assignment.

      # Calculate partitions for the original string (no changes made).
      max_p = self._count_partitions_core(s_list, k, n)

      # Iterate through all possible single character changes.
      for i in range(n):  # Index 'i' in s_list to change.
          original_char_at_i = s_list[i] # Store original char to backtrack later.
          
          for char_code in range(26): # Iterate 'a' through 'z' for the new character.
              new_char = chr(ord('a') + char_code)
              
              if new_char == original_char_at_i:
                  # If the new character is the same as the original at s_list[i],
                  # this modification results in the original string s_list.
                  # The partition count for the original string is already stored in max_p.
                  # So, we can skip this iteration.
                  continue
              
              s_list[i] = new_char # Perform the change.
              max_p = max(max_p, self._count_partitions_core(s_list, k, n))
              # Backtracking s_list[i] = original_char_at_i should be done after this inner loop.
          
          s_list[i] = original_char_at_i # Backtrack the change at index 'i' to restore s_list
                                         # before trying changes at the next index (i+1) or finishing.
      return max_p