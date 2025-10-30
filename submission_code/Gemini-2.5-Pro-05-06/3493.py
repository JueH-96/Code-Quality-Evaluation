class Solution:
  def maxOperations(self, s: str) -> int:
    ans = 0
    # num_ones_so_far counts all '1's encountered so far from the left.
    # These are '1's that are considered "active" or "floating" and will need to move
    # if they are followed by a '0'.
    num_ones_so_far = 0
    
    # This flag helps identify a '1' -> '0' transition.
    # It's true if the previous character processed was '1'.
    # Initialized to False, as there's no character before the first one.
    prev_char_was_one = False 
    
    for char_s_i in s:
      if char_s_i == '1':
        num_ones_so_far += 1
        prev_char_was_one = True  # Mark that the current character is '1'
      else: # char_s_i == '0'
        if prev_char_was_one: 
          # We've encountered a '0' immediately after a '1' (or a block of '1's).
          # This signifies the start of a new block of '0's (a "gap").
          # All 'num_ones_so_far' '1's accumulated to the left must eventually cross this gap.
          # Each such crossing by each of these '1's corresponds to one operation for that '1'.
          ans += num_ones_so_far
          
          # This specific jump operation moves a '1' past the entire block of '0's.
          # So, operations are only counted when a '1'-block meets a new '0'-block.
          # Set prev_char_was_one to False because the current character is '0'.
          # If the next character is also '0', it's part of the same '0'-block,
          # and we shouldn't add to 'ans' again for the same set of '1's jumping this same '0'-block.
          prev_char_was_one = False
          
    return ans