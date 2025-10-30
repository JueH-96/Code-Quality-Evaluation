import collections # Note: this import is not actually used and can be removed.

class Solution:
  """
  Finds the lexicographically smallest string obtainable by applying the operation exactly once.
  The operation involves selecting a non-empty substring and replacing each character
  with its preceding character ('a' becomes 'z').
  """
  def smallestString(self, s: str) -> str:
    """
    Applies the described operation once optimally to minimize the string lexicographically.

    Args:
      s: The input string consisting of lowercase English letters.

    Returns:
      The lexicographically smallest string after one operation.
    """
    
    n = len(s)
    # Convert the string to a list of characters for in-place modification.
    s_list = list(s)
    
    # Find the index k of the first character that is not 'a'.
    # The goal is to make the string lexicographically smallest. Changes should ideally happen
    # at the earliest possible index to have the largest impact.
    # Changing 'a' to 'z' increases the character lexicographically, which is undesirable unless
    # the entire string is 'a's.
    # Changing a character c > 'a' to c-1 decreases it, which is desirable.
    # Thus, we should look for the first non-'a' character to start our operation.
    k = 0
    while k < n and s_list[k] == 'a':
        k += 1
    
    # Case 1: If the entire string consists only of 'a's.
    if k == n:
        # Since we must perform the operation exactly once on a non-empty substring,
        # at least one 'a' must be changed to 'z'.
        # To minimize the resulting string lexicographically, we want the 'z' to appear
        # as late as possible. The optimal choice is to apply the operation to the
        # single character substring s[n-1..n-1]. This changes the last 'a' to 'z'.
        s_list[n - 1] = 'z'
        return "".join(s_list)
    
    # Case 2: The string contains at least one non-'a' character.
    # The first non-'a' character is at index k.
    
    # The optimal strategy is to start the substring modification at index k.
    # Starting earlier would change an 'a' to 'z' at an index i < k, resulting in a string
    # like "aa...az..." which is lexicographically larger than potentially starting at k
    # like "aa...a(c-1)...". Starting later than k misses the opportunity to reduce
    # the character at index k, resulting in a potentially larger string.
    
    # We should extend the substring modification from index k as long as the characters
    # are not 'a'. Modifying a character c > 'a' results in c-1, which makes the string
    # lexicographically smaller at that position. Modifying an 'a' results in 'z', which
    # makes the string lexicographically larger. Therefore, the optimal substring ends
    # just before the first 'a' encountered at or after index k.
    
    # Find the index p which marks the end of the contiguous block of non-'a' characters
    # starting from index k. p will be the index of the first 'a' at or after k,
    # or p = n if no 'a' is found in s[k..n-1].
    p = k
    while p < n and s_list[p] != 'a':
        p += 1
    
    # The substring to modify is s[k..p-1].
    # This range is guaranteed to be non-empty because we are in Case 2 (k < n),
    # which means s[k] is not 'a'. The while loop for p runs at least once for index k,
    # incrementing p to at least k+1. Thus the range [k, p-1] contains at least index k.
    
    # Apply the decrement operation to each character in the identified substring s[k..p-1].
    for i in range(k, p):
        # Since all characters s_list[i] for i in the range [k, p-1] are guaranteed 
        # to be greater than 'a', simply decrementing their ASCII value gives the 
        # correct preceding character in the alphabet. We don't need to handle the 'a' -> 'z'
        # wrap-around case within this loop.
        original_char_code = ord(s_list[i])
        s_list[i] = chr(original_char_code - 1)
        
    # Convert the modified list of characters back into a string and return it.
    return "".join(s_list)