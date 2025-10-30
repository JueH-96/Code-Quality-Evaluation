from typing import List

class Solution:
  def minOperations(self, nums: List[int], k: int) -> int:
    operations_count = 0
    for num in nums:
      if num < k:
        operations_count += 1
    return operations_count