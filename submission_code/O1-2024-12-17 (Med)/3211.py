class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        # We want to partition the array into contiguous blocks such that
        # the sums of those blocks form a non-decreasing sequence. The
        # final length is the number of blocks.
        
        # Greedy strategy:
        # 1. Maintain the sum of the "current block" (curr_sum).
        # 2. Keep track of the sum of the last finalized block (last_sum).
        # 3. As we iterate, keep adding elements to curr_sum until
        #    curr_sum >= last_sum. Once it is, we finalize that block,
        #    set last_sum = curr_sum, and reset curr_sum = 0.
        # 4. Each finalized block contributes +1 to the count.
        
        # This works because having a block sum as small as possible
        # makes it easier for the next block's sum to be >= it, thus
        # maximizing the number of blocks (and hence the final array length).
        
        count = 0
        last_sum = 0
        curr_sum = 0
        
        for x in nums:
            curr_sum += x
            # If the current block's sum is at least last finalized block's sum,
            # finalize the block here:
            if curr_sum >= last_sum:
                count += 1
                last_sum = curr_sum
                curr_sum = 0
        
        return count