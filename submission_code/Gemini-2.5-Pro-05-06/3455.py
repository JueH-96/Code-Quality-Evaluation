import collections

class Solution:
  def minimumLength(self, s: str) -> int:
    """
    Calculates the minimum length of the final string s after performing
    the described operations any number of times.

    The operation is:
    1. Choose an index i such that s[i] has at least one identical character
       to its left and at least one identical character to its right.
    2. Delete the closest character to the left of i that is equal to s[i].
    3. Delete the closest character to the right of i that is equal to s[i].

    Each operation involves a character s[i] (let's call it 'x') and two other
    occurrences of 'x', one to the left and one to the right. These two other
    occurrences are removed. The character s[i] itself is NOT removed.
    So, each operation reduces the count of character 'x' by 2.

    This process can be repeated for character 'x' as long as its count is >= 3.
    The condition for choosing index i (s[i] must have identical characters on
    both left and right) means that s[i] cannot be the first or last occurrence
    of its type. This requires at least 3 occurrences of that character type
    for an operation to be possible.

    - If the initial count of 'x' (let's denote k_x) is odd:
      - If k_x = 1: No operations possible. 1 'x' remains.
      - If k_x = 3: One operation. Count becomes 3-2=1. 1 'x' remains.
      - If k_x = 5: Two operations. Count becomes 5-2-2=1. 1 'x' remains.
      In general, if k_x is odd (and positive), 1 'x' will remain.
    - If the initial count of 'x' (k_x) is positive and even:
      - If k_x = 2: No operations possible. 2 'x's remain.
      - If k_x = 4: One operation. Count becomes 4-2=2. 2 'x's remain.
      - If k_x = 6: Two operations. Count becomes 6-2-2=2. 2 'x's remain.
      In general, if k_x is positive and even, 2 'x's will remain.
    - If k_x = 0 (character not in string), 0 'x's remain.

    The operations on one character type do not affect the counts of other
    character types, nor do they prevent or enable operations for other types.
    Therefore, we can calculate the final count for each character type
    independently based on its initial count. The sum of these final counts
    gives the minimum final length of the string.
    """
    
    # Count frequencies of each character in the string s.
    # collections.Counter is efficient for this: O(length of s).
    counts = collections.Counter(s)
    
    final_length = 0
    # Iterate through the counts of characters that are present in s.
    # counts.values() gives an iterator over these counts.
    # Each char_count here will be >= 1 because Counter only stores
    # entries for characters present in the input string.
    for char_count in counts.values():
      if char_count % 2 == 1:  # If the count is odd
        final_length += 1  # 1 character of this type will remain
      else:  # If the count is even (and positive)
        final_length += 2  # 2 characters of this type will remain
            
    return final_length