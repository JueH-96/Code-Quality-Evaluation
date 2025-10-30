import collections

class Solution:
  def findValidPair(self, s: str) -> str:
    n = len(s)
    # According to constraints: 2 <= s.length <= 100.
    # So, n is guaranteed to be at least 2.

    # 1. Pre-calculate digit frequencies in the string s.
    # collections.Counter is efficient for this.
    # Example: s = "2523533", counts will be {'2': 2, '5': 2, '3': 3}
    counts = collections.Counter(s)

    # 2. Iterate through all adjacent pairs of digits in s from left to right.
    for i in range(n - 1): # i ranges from 0 to n-2
      d1_char = s[i]      # The first digit in the pair
      d2_char = s[i+1]    # The second digit in the pair

      # Condition 1: The first digit is not equal to the second.
      # (This is one of the requirements for a valid pair)
      if d1_char == d2_char:
        continue # Skip if digits are the same, move to the next pair

      # Convert character digits to their integer numeric values.
      # These numeric values are the target counts for each digit.
      # Example: if d1_char is '2', then d1_target_count is 2.
      # This means digit '2' must appear 2 times in the string s for it to be part of a valid pair.
      # Constraints state digits are '1' to '9', so int() conversion is safe and results in 1-9.
      d1_target_count = int(d1_char)
      d2_target_count = int(d2_char)
      
      # Retrieve the actual counts of these digits from our pre-calculated map.
      # counts[d1_char] gives how many times d1_char (e.g., '2') appears in s.
      # counts[d2_char] gives how many times d2_char (e.g., '3') appears in s.
      # Note: Since d1_char and d2_char are characters taken from s, they are guaranteed 
      # to exist as keys in the 'counts' Counter object.
      actual_count_d1 = counts[d1_char]
      actual_count_d2 = counts[d2_char]

      # Condition 2: Each digit in the pair appears in s exactly as many times as its numeric value.
      # Both d1 and d2 must satisfy this condition.
      if actual_count_d1 == d1_target_count and actual_count_d2 == d2_target_count:
        # If both this condition and the (d1_char != d2_char) condition are met,
        # we have found the first valid pair (due to left-to-right iteration).
        # Concatenate the characters to form the result string (e.g., "2" + "3" = "23").
        return d1_char + d2_char
      
    # 3. If the loop completes, it means no valid pair was found in the string s.
    # In this case, return an empty string as per the problem specification.
    return ""