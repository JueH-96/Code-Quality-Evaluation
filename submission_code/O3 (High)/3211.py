from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        """
        Greedy idea
        ------------
        We want to split the original array into as many contiguous blocks as
        possible, such that the sequence of the block–sums is non-decreasing.
        Because every element is positive, the following greedy strategy is
        optimal:

        • keep reading the array from left to right;
        • accumulate the current block sum;
        • the moment the accumulated sum becomes at least as large as the
          previous block-sum we **close** the current block (this is the
          smallest possible block that keeps the whole sequence valid);
        • start a new block and repeat.

        The algorithm is optimal because no valid partition can place a cut
        earlier than the greedy one – the sum before reaching `prev_sum` would
        necessarily be smaller than `prev_sum` and would violate the
        non-decreasing requirement.

        Implementation
        --------------
        prev_sum ... sum of the last finished block (0 for the first one)
        cur_sum  ... running sum of the current block
        ans      ... number of finished blocks (= desired length)

        Complexity
        ----------
        Time  : O(n)
        Memory: O(1)
        """
        prev_sum = 0      # sum of the last finalised block
        cur_sum  = 0      # running sum of the current block
        ans      = 0      # number of blocks obtained so far
        
        for x in nums:
            cur_sum += x
            if cur_sum >= prev_sum:        # can close the current block
                ans += 1
                prev_sum = cur_sum
                cur_sum  = 0               # start a new block
                
        # any leftover (cur_sum < prev_sum) is merged with the last block,
        # so it does not change the count `ans`.
        return ans