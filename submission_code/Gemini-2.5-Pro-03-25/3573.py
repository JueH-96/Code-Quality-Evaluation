import collections

class Solution:
  """
  Solves the problem of finding the number of valid substrings in word1.
  A substring is valid if its character counts are sufficient to form word2 as a prefix after rearrangement.
  This means for every character 'c', the count of 'c' in the substring must be >= the count of 'c' in word2.
  """
  def validSubstringCount(self, word1: str, word2: str) -> int:
    """
    Finds the total number of valid substrings of word1 using a sliding window approach.

    Args:
      word1: The main string to find substrings within. Its length is n.
      word2: The string whose character counts define the validity condition. Its length is m.

    Returns:
      The total count of valid substrings in word1.
    """
    n = len(word1)
    m = len(word2)

    # Constraints state m >= 1, so word2 is never empty.

    # 1. Calculate target character counts for word2 using an array.
    # This array stores the minimum required frequency for each character 'a' through 'z'.
    target_counts_arr = [0] * 26
    for char_code in word2.encode('ascii'): # Using encode().ascii can be slightly faster
        target_counts_arr[char_code - ord('a')] += 1

    # Determine the number of distinct character types that have a requirement (count > 0).
    num_required_types = 0
    for i in range(26):
        if target_counts_arr[i] > 0:
            num_required_types += 1

    # If word2 required no characters (e.g., contains only characters with 0 required count,
    # although this scenario seems unlikely given m>=1 unless defined differently),
    # every substring would be valid. Under standard interpretation and m>=1, num_required_types >= 1.
    if num_required_types == 0:
         # This case should not be reachable under standard interpretation of the problem
         # and constraints. If it were possible, all n*(n+1)//2 substrings would be valid.
         return n * (n + 1) // 2


    # 2. Initialize sliding window variables
    left_ptr = 0  # Left pointer 'i' of the sliding window [left_ptr, right_ptr]
    total_valid_substrings = 0 # Accumulates the total count of valid substrings
    current_counts_arr = [0] * 26 # Character counts for the current window word1[left_ptr...right_ptr]
    satisfied_types_count = 0 # Count of character types meeting their target count in the current window

    # 3. Iterate through word1 with the right pointer 'right_ptr' (representing 'j')
    for right_ptr in range(n):
        # Expand the window by including word1[right_ptr]
        char_code = ord(word1[right_ptr])
        char_idx = char_code - ord('a')
        current_counts_arr[char_idx] += 1

        # Check if adding this character helps satisfy the requirement for its type.
        # We increment satisfied_types_count only when a type *first* meets its requirement.
        if target_counts_arr[char_idx] > 0 and current_counts_arr[char_idx] == target_counts_arr[char_idx]:
            # This specific character type just met its required count
            satisfied_types_count += 1

        # 4. Shrink the window from the left while it remains valid.
        # The window word1[left_ptr...right_ptr] is valid if all required character types
        # have sufficient counts, which means satisfied_types_count equals num_required_types.
        while satisfied_types_count == num_required_types:
            # Since word1[left_ptr...right_ptr] is valid, all substrings ending at right_ptr
            # with starting index k <= left_ptr are also valid. We need to find how many such k exist.
            # By shrinking the window (incrementing left_ptr), we find the boundary.

            # Consider the character to be removed from the left of the window
            char_to_remove_code = ord(word1[left_ptr])
            char_to_remove_idx = char_to_remove_code - ord('a')

            # Check if removing this character will cause its type to become unsatisfied.
            # This happens if the current count for this type is exactly the target count.
            if target_counts_arr[char_to_remove_idx] > 0 and current_counts_arr[char_to_remove_idx] == target_counts_arr[char_to_remove_idx]:
                # The count was exactly the target, so removing one will make it fall below the requirement.
                satisfied_types_count -= 1

            # Update the count for the removed character in the window
            current_counts_arr[char_to_remove_idx] -= 1

            # Move the left pointer of the window to the right
            left_ptr += 1

        # 5. Add the contribution of the current endpoint 'right_ptr' to the total count.
        # After the while loop finishes (or if it never ran), `left_ptr` represents the first
        # index such that the window word1[left_ptr...right_ptr] is *not* valid.
        # This implies that all windows word1[k...right_ptr] where 0 <= k < left_ptr *are* valid.
        # The number of such valid starting indices k is exactly `left_ptr`.
        # For each `right_ptr`, we add the count of valid starting positions found.
        total_valid_substrings += left_ptr

    return total_valid_substrings