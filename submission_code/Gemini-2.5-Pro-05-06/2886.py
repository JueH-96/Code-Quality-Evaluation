import collections

class Solution:
  def finalString(self, s: str) -> str:
    text_deque = collections.deque()
    # This flag tracks if the current "view" of the string is effectively reversed.
    # If True, new characters are added to what would be the "start" of the string
    # if it were laid out linearly from left to right. This means prepending to the deque.
    # If False, new characters are added to the "end" of the string, meaning appending to the deque.
    is_view_reversed = False 
    
    for char_typed in s:
      if char_typed == 'i':
        # Typing 'i' reverses the string on screen. We simulate this by flipping
        # our 'is_view_reversed' flag. This changes how subsequent characters are added.
        is_view_reversed = not is_view_reversed
      else:
        if is_view_reversed:
          # If the current view is reversed, appending a character effectively means
          # adding it to the left end of our deque.
          text_deque.appendleft(char_typed)
        else:
          # If the current view is normal, append character to the right end of deque.
          text_deque.append(char_typed)
          
    # After processing all characters, if the final view is reversed,
    # the deque's current order (left-to-right) is the reverse of the final string.
    # So, we must reverse the deque itself to get the characters in the correct final order.
    if is_view_reversed:
      text_deque.reverse() # In-place O(N) reversal for deque
      
    return "".join(text_deque)