import math
from typing import List

class Solution:
  """
  Solves the Longest Valid Substring problem using a sliding window approach.
  A substring is valid if none of its substrings are present in the forbidden list.
  We aim to find the length of the longest such valid substring within the given word.
  """
  def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
    """
    Finds the length of the longest substring of 'word' that does not contain
    any string from 'forbidden' as a substring.

    Args:
        word: The main string.
        forbidden: A list of forbidden substrings.

    Returns:
        The length of the longest valid substring.
    
    Constraints:
        1 <= word.length <= 10^5
        word consists only of lowercase English letters.
        1 <= forbidden.length <= 10^5
        1 <= forbidden[i].length <= 10
        forbidden[i] consists only of lowercase English letters.

    Complexity:
        Time: O(N * max_k^2 + L), where N = len(word), L = total length of forbidden strings, 
              max_k = max length of forbidden strings (<= 10).
              Since max_k is a small constant (<= 10), the complexity is effectively O(N + L).
        Space: O(L) for storing the forbidden set, where L is the sum of lengths of strings in forbidden.
    """

    # Create a set from the forbidden list for efficient O(1) average time lookup.
    # The time complexity for creating the set is proportional to the total number of characters
    # in all forbidden strings, let's denote this by L.
    # The space complexity is also O(L).
    forbidden_set = set(forbidden)

    # Determine the maximum length of a forbidden string. 
    # According to constraints, 1 <= forbidden[i].length <= 10, so max_k will be between 1 and 10.
    # This step takes O(M) time if M is the number of forbidden strings, assuming we have lengths, or O(L) if we iterate through strings.
    max_k = 0
    # The constraint 1 <= forbidden.length ensures the list is not empty.
    for f in forbidden: 
         max_k = max(max_k, len(f))
    # Alternatively, we could just use max_k = 10 based on the constraint.

    n = len(word)
    max_len = 0
    
    # 'min_valid_left' represents the smallest possible starting index 'i' such that
    # the substring word[i : right+1] is guaranteed to be valid based on the forbidden strings
    # checked up to the current 'right' index.
    # It effectively defines the left boundary of the longest valid substring ending at 'right'.
    min_valid_left = 0

    # Iterate through the word using 'right' as the right boundary of the sliding window. O(N) iterations.
    for right in range(n):
        # Check potential forbidden substrings ending exactly at index 'right'.
        # We only need to consider substrings with length up to max_k, because any forbidden string
        # must have length at most max_k.
        # The starting index 'k' for such substrings ranges from max(0, right - max_k + 1) to right.
        
        # Optimization: We only need to check substrings that start at or after the current 'min_valid_left'.
        # If a forbidden substring started at k' < min_valid_left, it means 'min_valid_left' was already
        # updated to be at least k' + 1 in a previous iteration (for some smaller 'right').
        # So, checking starting from max(min_valid_left, right - max_k + 1) is sufficient.
        start_check_idx = max(min_valid_left, right - max_k + 1)
        
        # This inner loop iterates at most max_k times. O(max_k) iterations.
        for k in range(start_check_idx, right + 1):
            # Extract the substring ending at 'right' starting from 'k'.
            # String slicing in Python takes O(length of slice) time, which is O(max_k) here.
            substring = word[k : right + 1]
            
            # Check if this substring exists in the forbidden set.
            # Hash lookup in a set takes O(length of string) on average, which is O(max_k) here.
            # The total cost for this inner loop iteration is O(max_k).
            # The total cost for the whole inner loop per 'right' iteration is O(max_k * max_k).
            if substring in forbidden_set:
                # If word[k:right+1] is a forbidden substring, it means any valid substring
                # ending at 'right' must start *after* index 'k'. 
                # So, the minimum valid starting index must be at least k + 1.
                # We update min_valid_left to be the maximum of its current value and k + 1.
                # We need to find the largest k that makes word[k:right+1] forbidden, 
                # as this imposes the most restrictive (largest) lower bound on the starting index.
                # The loop structure ensures we consider all relevant k and find the max necessary update.
                min_valid_left = max(min_valid_left, k + 1)

        # After checking all relevant substrings ending at 'right', 'min_valid_left' holds the 
        # smallest valid start index for a substring ending at 'right'.
        # The length of the longest valid substring ending at 'right' is (right - min_valid_left + 1).
        # If min_valid_left > right, this length calculation correctly yields a value <= 0.
        current_len = right - min_valid_left + 1
        
        # Update the overall maximum length found so far across all possible end points 'right'.
        max_len = max(max_len, current_len)

    # Return the maximum length found.
    return max_len