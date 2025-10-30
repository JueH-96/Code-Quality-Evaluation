class Solution:
  def getSmallestString(self, s: str, k: int) -> str:
    """
    Finds the lexicographically smallest string t such that distance(s, t) <= k.

    The distance between two characters is their minimum distance in a cyclic
    order ('a'...'z'). For example, distance('a', 'z') is 1.
    The distance between two strings is the sum of character distances.

    Args:
      s: The original string.
      k: The maximum allowed distance.

    Returns:
      The lexicographically smallest string t.
    """

    # Helper function to calculate the distance from a character value (0-25) to 'a' (value 0).
    # A character's value is its 0-indexed position in the alphabet, e.g., 'a' -> 0, 'b' -> 1, ...
    def dist_to_a_val(char_val: int) -> int:
        # Distance to 'a' can be achieved by moving "down" (char_val steps)
        # or "up" (26 - char_val steps, wrapping around z->a).
        # e.g., for 'c' (val 2): min(2, 26-2) = 2.
        # e.g., for 'y' (val 24): min(24, 26-24) = 2.
        # 'a' (val 0) itself: min(0, 26-0) = 0.
        return min(char_val, 26 - char_val)

    n = len(s)
    # Convert string to a list of characters for easier modification.
    # Initialize t_chars with characters from s. If k runs out early,
    # the remaining characters in t_chars will correctly be from the original s.
    t_chars = list(s)

    for i in range(n):
        if k == 0:
            # No budget left, no more changes can be made.
            # The rest of t_chars will remain as they were in s.
            break

        # Get the 0-25 value of the current character s[i].
        # t_chars[i] initially holds s[i]. We are deciding what t_chars[i] should become.
        original_char_val = ord(t_chars[i]) - ord('a')
        
        # Calculate the cost to change s[i] to 'a'.
        # If s[i] is already 'a' (original_char_val is 0), cost_to_make_a is 0.
        # In this case, it will fall into the first 'if' block below.
        # t_chars[i] will be set to 'a' (no change), and k will be reduced by 0 (no change).
        # This behavior is correct.
        cost_to_make_a = dist_to_a_val(original_char_val)

        if cost_to_make_a <= k:
            # We can afford to change s[i] to 'a'.
            # This is the lexicographically best choice for t_chars[i].
            t_chars[i] = 'a'
            k -= cost_to_make_a
        else:
            # We cannot afford to change s[i] to 'a'. (cost_to_make_a > k)
            # We must use all remaining budget k to make t_chars[i]
            # as lexicographically small as possible.
            # This means changing s[i] to the character that is k steps
            # "down" from s[i] (i.e., towards 'a' by decreasing its ASCII value).
            # The cost of this operation will be exactly k.
            
            # If this branch is reached, then original_char_val > k.
            # So, ord(t_chars[i]) - k will result in an ASCII value for a character >= 'a'.
            
            new_char_ascii = ord(t_chars[i]) - k
            t_chars[i] = chr(new_char_ascii)
            k = 0  # All budget is consumed.
            # The loop will break in the next iteration due to k == 0 check at the top.

    return "".join(t_chars)