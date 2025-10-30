from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Option 1: Build a valid subsequence where every adjacent sum is even.
        # For two numbers a and b, (a+b) % 2 is 0 when a and b have the same parity.
        # Thus, if we pick a subsequence containing only even or only odd numbers,
        # every pair’s sum will be even.
        evenCount = sum(1 for x in nums if x % 2 == 0)
        oddCount  = len(nums) - evenCount
        same_parity_max = max(evenCount, oddCount)
        
        # Option 2: Build a valid subsequence where every adjacent sum is odd.
        # For two numbers a and b, (a+b) % 2 is 1 when a and b have different parity.
        # Thus, a valid subsequence in this case must alternate in parity.
        # We can choose a subsequence greedily by deciding which parity to start with.
        def alt_sequence(nums: List[int], start_parity: int) -> int:
            count = 0
            needed = start_parity
            for num in nums:
                if num % 2 == needed:
                    count += 1
                    needed = 1 - needed  # flip the needed parity for the next element
            return count
        
        alt_max = max(alt_sequence(nums, 0), alt_sequence(nums, 1))
        
        # The answer is the best we can do either by picking numbers of the same
        # parity (which yields even sums) or by alternating parities (yielding odd sums).
        return max(same_parity_max, alt_max)