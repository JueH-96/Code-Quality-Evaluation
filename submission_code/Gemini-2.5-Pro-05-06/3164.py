from typing import List

class Solution:
  def lastVisitedIntegers(self, words: List[str]) -> List[int]:
    nums_seen: List[int] = []
    # k_consecutive_prevs stores the number of consecutive "prev" strings encountered.
    # It's reset to 0 when an integer is encountered.
    # It's incremented when a "prev" is encountered.
    k_consecutive_prevs: int = 0
    
    result_list: List[int] = []
    
    for current_word in words:
      if current_word == "prev":
        k_consecutive_prevs += 1
        
        # If k_consecutive_prevs is greater than the total number of integers seen so far (len(nums_seen)),
        # then the last visited integer is -1.
        if k_consecutive_prevs > len(nums_seen):
          result_list.append(-1)
        else:
          # The problem asks for the element at (k_consecutive_prevs - 1)-th index of nums_reverse.
          # If nums_seen has L elements [N_0, ..., N_{L-1}],
          # nums_reverse is [N_{L-1}, ..., N_0].
          # The element at index (k_consecutive_prevs - 1) in nums_reverse
          # is N_{L-1-(k_consecutive_prevs-1)} = N_{L-k_consecutive_prevs}.
          # This is equivalent to nums_seen[L-k_consecutive_prevs].
          # In Python, this is nums_seen[-k_consecutive_prevs].
          result_list.append(nums_seen[-k_consecutive_prevs])
      else:
        # The current word is an integer string.
        # Reset the count of consecutive "prev" strings.
        k_consecutive_prevs = 0
        # Add the integer to our list of seen numbers.
        num = int(current_word)
        nums_seen.append(num)
        
    return result_list