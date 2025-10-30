import math
import heapq
import collections
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    """
    Calculates the total number of possible original strings Alice might have intended to type,
    given the final output string `word` where at most one character might have been
    typed multiple times due to a long key press.
    """
    def possibleStringCount(self, word: str) -> int:
        """
        Args:
            word: The final string displayed on Alice's screen. Consists of lowercase English letters. Length is between 1 and 100.

        Returns:
            The total number of possible original strings.
        """
        n = len(word)
        # Constraints state 1 <= word.length <= 100, so n is guaranteed to be at least 1.

        # Initialize count with 1. This accounts for the possibility that
        # the typed word 'word' is exactly the intended string, meaning no 
        # accidental long press occurred.
        total_count = 1

        i = 0
        # Iterate through the string to identify runs of consecutive identical characters.
        while i < n:
            # Identify the character starting the current run.
            current_char = word[i]
            
            # Find the end index 'j' of the current run.
            # 'j' will point to the first character after the run.
            j = i
            while j < n and word[j] == current_char:
                j += 1
            
            # Calculate the length of the run.
            run_length = j - i

            # If the run length is 2 or more, this run could potentially be
            # the result of the single allowed long key press event.
            # A run of length 'run_length' (>= 2) could have been formed from an original run 
            # of 'current_char' with length k, where 1 <= k < run_length.
            # If this specific run was the site of the long press, there are 
            # 'run_length - 1' possible lengths for the original run (1, 2, ..., run_length - 1).
            # Each of these corresponds to a unique possible original string.
            if run_length >= 2:
                # Add the number of possibilities originating from this run being the long press site.
                total_count += run_length - 1

            # Move the index 'i' to the beginning of the next run for the next iteration.
            i = j

        # The final 'total_count' sums the one possibility of no long press 
        # and all the possibilities where exactly one long press occurred at any valid position.
        return total_count