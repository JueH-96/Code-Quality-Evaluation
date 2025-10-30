class Solution:
  def getSmallestString(self, s: str) -> str:
    n = len(s)
    # Convert string to list of characters because strings are immutable in Python,
    # and we might need to perform a swap.
    s_chars = list(s)

    # Iterate through all possible adjacent pairs (s_chars[i], s_chars[i+1]).
    # The loop runs for i from 0 to n-2.
    # Example: if n=2, range(n-1) is range(1), so loop runs for i=0. s_chars[0], s_chars[1] checked.
    # Example: if n=3, range(n-1) is range(2), so loop runs for i=0,1. 
    #          (s_chars[0],s_chars[1]) then (s_chars[1],s_chars[2]) are checked.
    for i in range(n - 1):
      # Get the characters at s_chars[i] and s_chars[i+1]
      char_i = s_chars[i]
      char_i_plus_1 = s_chars[i+1]
      
      # Get their integer values to check parities.
      # s consists only of digits, so int() conversion is safe.
      val_i = int(char_i)
      val_i_plus_1 = int(char_i_plus_1)

      # Check if the two digits have the same parity.
      # Parity is (value % 2). Same parity means (val1 % 2) == (val2 % 2).
      if (val_i % 2) == (val_i_plus_1 % 2):
        # Parities are the same. A swap between char_i and char_i_plus_1 is permissible.
        
        # Now, we need to check if performing this swap would result in a
        # lexicographically smaller string. This happens if char_i_plus_1
        # is smaller than char_i.
        # Note: Character comparison ('0' < '1', etc.) works correctly for digits.
        if char_i_plus_1 < char_i:
          # This is a "beneficial" swap: it's allowed AND it makes the string smaller.
          # Perform the swap on our list of characters.
          s_chars[i], s_chars[i+1] = char_i_plus_1, char_i
          
          # The problem states we can perform AT MOST one swap.
          # We are looking for the lexicographically smallest string.
          # By iterating from left to right (increasing i), the first beneficial
          # swap we find will modify the string at the earliest possible position (index i)
          # to make it smaller (s_chars[i] is replaced by a smaller character char_i_plus_1).
          # This strategy guarantees the lexicographically smallest result among
          # all strings formed by one swap. The resulting string is also guaranteed to be
          # smaller than the original string.
          # Therefore, we perform this one swap and this is our final answer.
          return "".join(s_chars)
          
    # If the loop completes, it means no "beneficial" swap was found.
    # A "beneficial" swap is one where char_i_plus_1 < char_i AND they have the same parity.
    # If no such swap exists, any permissible swap would either:
    #  1. Make the string lexicographically larger (if char_i_plus_1 > char_i).
    #  2. Keep the string the same (if char_i_plus_1 == char_i).
    # In this scenario, not performing any swap is the best strategy because we want the
    # lexicographically smallest string.
    # The original string `s` is already the lexicographically smallest string achievable.
    return s