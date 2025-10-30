import math # This import is not strictly necessary for the final code but was used during thought process.

class Solution:
  """
  Solves the problem of minimizing the length of the longest substring 
  of identical characters in a binary string by performing at most numOps flips.
  Uses binary search on the answer (minimum maximum length) and checks 
  feasibility by simulating two greedy flipping strategies.
  """
  def minLength(self, s: str, numOps: int) -> int:
    """
    Finds the minimum possible length of the longest substring of identical 
    characters after performing at most numOps flips.

    Args:
        s: The input binary string.
        numOps: The maximum number of allowed flips.

    Returns:
        The minimum possible length of the longest substring.
    """
    n = len(s)

    def get_max_run(arr, char):
      """
      Calculates the maximum length of consecutive runs of 'char' in 'arr'.
      'arr' should be an iterable (e.g., list or string).
      """
      max_run = 0
      current_run = 0
      for c in arr:
        if c == char:
          current_run += 1
        else:
          # End of a run of 'char', update max_run if needed
          max_run = max(max_run, current_run)
          current_run = 0 # Reset for the next potential run
      # Check the last run after the loop finishes
      max_run = max(max_run, current_run)
      return max_run

    def check_char(char_to_flip, k, current_s_str, numOps_limit):
      """
      Checks if it's possible to make the maximum run length <= k by performing 
      at most numOps_limit flips, where ONLY 'char_to_flip' is flipped to the other character.
      This function simulates the greedy flipping strategy: whenever a run of 
      'char_to_flip' reaches length k+1, flip the last character of that run.
      It returns True if this strategy succeeds within the flip limit AND the 
      resulting string has no runs of the *other* character longer than k.
      Otherwise, it returns False.

      Args:
          char_to_flip: The character ('0' or '1') that we are allowed to flip.
          k: The target maximum run length.
          current_s_str: The original string.
          numOps_limit: The maximum number of flips allowed for this strategy.

      Returns:
          True if the condition can be met using this strategy, False otherwise.
      """
      m = len(current_s_str)
      other_char = '1' if char_to_flip == '0' else '0'
      # Create a mutable copy of the string to simulate flips
      s_list_after_flips = list(current_s_str)
      flips_made = 0
      consecutive_count = 0

      # Iterate through the string and apply greedy flips
      for i in range(m):
        # Check the character in the potentially modified list
        char_at_i = s_list_after_flips[i]

        if char_at_i == char_to_flip:
          consecutive_count += 1
        else:
          # Reset count if the character is not the one we are tracking for runs
          consecutive_count = 0

        # If a run of char_to_flip reaches length k+1
        if consecutive_count == k + 1:
          # We need to perform a flip
          flips_made += 1
          # Check if we have exceeded the allowed number of flips
          if flips_made > numOps_limit:
            return False # Cannot achieve the goal with this strategy and limit

          # Perform the flip on the last character of the k+1 run
          s_list_after_flips[i] = other_char
          # Reset the consecutive count because the character at index i is now flipped,
          # effectively breaking the run ending at i.
          consecutive_count = 0

      # After iterating and flipping, we know flips_made <= numOps_limit.
      # The flips ensure that runs of char_to_flip are at most k.
      # Now, we must verify that the runs of the *other* character 
      # (which might have increased due to flips) also do not exceed k.
      max_other_run = get_max_run(s_list_after_flips, other_char)

      # The condition is met if the longest run of the other character is also <= k
      return max_other_run <= k

    def can(k):
      """
      Checks if it's possible to make the maximum run length at most k
      using at most numOps flips. It assumes that if a solution exists,
      at least one of the two greedy single-character-flip strategies must work.
      
      Args:
          k: The target maximum run length to check.

      Returns:
          True if max run length k is achievable, False otherwise.
      """
      # Check strategy 1: Greedily flip '0's to break runs longer than k.
      # If this works (uses <= numOps flips and resulting '1' runs are also <= k), return True.
      if check_char('0', k, s, numOps):
        return True

      # Check strategy 2: Greedily flip '1's to break runs longer than k.
      # If this works (uses <= numOps flips and resulting '0' runs are also <= k), return True.
      if check_char('1', k, s, numOps):
        return True

      # If neither pure greedy strategy works, we conclude it's not possible to achieve k.
      return False

    # --- Binary Search for the minimum k ---
    # The minimum possible answer is 1 (e.g., "0101").
    # The maximum possible answer is n (e.g., "0000" with numOps=0).
    low = 1
    high = n
    ans = n # Initialize answer to the worst-case scenario

    while low <= high:
      mid = low + (high - low) // 2 # Calculate the midpoint length to test
      # Check if a maximum run length of 'mid' is achievable
      if can(mid):
        # If yes, 'mid' is a possible answer. Store it and try for smaller lengths.
        ans = mid
        high = mid - 1
      else:
        # If no, the minimum achievable length must be greater than 'mid'.
        low = mid + 1

    # The loop terminates when low > high, and 'ans' holds the smallest k for which can(k) was True.
    return ans