class Solution:
  def clearDigits(self, s: str) -> str:
    # We use a list of characters to build the result string.
    # This list effectively acts as a stack.
    res_chars = []
    
    # Iterate through each character of the input string.
    for char_val in s:
      if char_val.isdigit():
        # If the character is a digit:
        # According to the problem, this digit must be removed along with
        # the closest non-digit character to its left.
        # In our left-to-right scan, the characters currently in `res_chars`
        # are all non-digits that are to the left of `char_val` and have not
        # yet been removed. The last one added (`res_chars[-1]`) is the
        # closest one.
        
        # The problem guarantees "it is possible to delete all digits".
        # This implies that `res_chars` will not be empty when a digit is
        # encountered, because there must be a non-digit to its left to remove.
        # So, we pop from `res_chars`.
        # The check `if res_chars:` is for defensive programming;
        # strictly based on problem constraints, it should always be true
        # that res_chars is not empty here because it will only contain non-digits.
        if res_chars:
            res_chars.pop()
        # The digit `char_val` itself is not added to `res_chars` because it's "removed".
      else:
        # If the character is not a digit (i.e., a letter):
        # Add it to our list. It might be removed later if a digit appears to its right.
        res_chars.append(char_val)
        
    # Join the characters in res_chars to form the final string.
    return "".join(res_chars)