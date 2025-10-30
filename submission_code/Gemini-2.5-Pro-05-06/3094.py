import collections
from typing import List # Required for type hinting List[int] in Python < 3.9

class Solution:
  def minOperations(self, nums: List[int]) -> int:
    counts = collections.Counter(nums)
    
    total_operations = 0
    
    # Iterate through the frequencies of each distinct number.
    # The actual value of the number doesn't influence the calculation, only its count.
    for count in counts.values():
      if count == 1:
        # If any number appears exactly once, it's impossible to remove it.
        # Operations require deleting 2 or 3 identical elements at a time.
        return -1
      
      # For a number appearing 'count' times (where count >= 2):
      # We need to express 'count' as a sum of 2s and 3s: count = 2*op2 + 3*op3.
      # We want to minimize the number of operations: op2 + op3.
      # It can be shown that this is achieved by maximizing op3 (number of "delete 3" operations).
      # The formula (count + 2) // 3 gives this minimum.
      # This is equivalent to math.ceil(count / 3.0).
      #
      # Detailed breakdown:
      # - If count % 3 == 0 (e.g., count = 6):
      #   Use count/3 operations of type "delete 3". Ops = count / 3.
      #   (6 + 2) // 3 = 8 // 3 = 2. Correct.
      # - If count % 3 == 1 (e.g., count = 4, count = 7):
      #   This implies count = 3k + 1. Since count >= 2, k >= 1.
      #   Cannot clear with only "delete 3"s.
      #   We take 3(k-1) elements using (k-1) "delete 3" ops, leaving 4 elements.
      #   These 4 elements are cleared with two "delete 2" ops.
      #   Total ops = (k-1) + 2 = k+1.
      #   count = 4: (4 + 2) // 3 = 6 // 3 = 2. (k=1, ops = 1+1=2). Correct.
      #   count = 7: (7 + 2) // 3 = 9 // 3 = 3. (k=2, ops = 2+1=3). Correct.
      # - If count % 3 == 2 (e.g., count = 2, count = 5):
      #   This implies count = 3k + 2.
      #   Take 3k elements using k "delete 3" ops, leaving 2 elements.
      #   These 2 elements are cleared with one "delete 2" op.
      #   Total ops = k + 1.
      #   count = 2: (2 + 2) // 3 = 4 // 3 = 1. (k=0, ops = 0+1=1). Correct.
      #   count = 5: (5 + 2) // 3 = 7 // 3 = 2. (k=1, ops = 1+1=2). Correct.
      
      operations_for_this_count = (count + 2) // 3
      total_operations += operations_for_this_count
      
    return total_operations