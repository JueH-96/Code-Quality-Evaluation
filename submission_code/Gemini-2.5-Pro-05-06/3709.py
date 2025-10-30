class Solution:
  def hasSpecialSubstring(self, s: str, k: int) -> bool:
    n = len(s)

    # According to constraints: 1 <= k <= s.length.
    # So, n >= k. The loop range n - k + 1 will be at least 1.
    # (e.g., if n = k, loop is range(1), so i = 0).
    
    for i in range(n - k + 1):
      # Current potential substring starts at index i and has length k.
      # This substring is s[i : i+k].
      
      # Condition 1: The substring consists of only one distinct character.
      # We take the first character s[i] as the character that should be repeated.
      char_of_sub = s[i]
      
      # Check if all characters from s[i+1] to s[i+k-1] are also char_of_sub.
      # If k=1, this loop for j (from 1 to 0) doesn't run, is_single_char remains True.
      is_single_char = True
      for j in range(1, k): # j iterates from 1 up to k-1.
                            # s[i+j] accesses s[i+1], ..., s[i+k-1].
        if s[i+j] != char_of_sub:
          is_single_char = False
          break
      
      if not is_single_char:
        # This substring s[i : i+k] doesn't meet condition 1.
        # Move to the next potential starting position i.
        continue

      # Condition 1 is met. The character for the substring is char_of_sub.

      # Condition 2: If there is a character immediately before the substring, 
      # it must be different from char_of_sub.
      # This condition fails if (i > 0 AND s[i-1] == char_of_sub).
      # If i == 0, there's no character before, so this condition is met by default.
      if i > 0: # Check only if a character exists before the substring
        if s[i-1] == char_of_sub:
          # Character before is the same as char_of_sub. Fails condition 2.
          continue # Move to the next starting position i.
      
      # Condition 3: If there is a character immediately after the substring,
      # it must also be different from char_of_sub.
      # This condition fails if (i+k < n AND s[i+k] == char_of_sub).
      # If i+k == n, the substring is at the end of s, so no character after.
      # This condition is met by default in that case.
      if i + k < n: # Check only if a character exists after the substring
        if s[i+k] == char_of_sub:
          # Character after is the same as char_of_sub. Fails condition 3.
          continue # Move to the next starting position i.
      
      # If we reach here, all three conditions are met for the substring s[i : i+k].
      return True
            
    # If the loop finishes, no such substring was found.
    return False