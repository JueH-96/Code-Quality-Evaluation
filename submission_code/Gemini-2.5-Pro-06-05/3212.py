from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # A partition is good if for any number, all its occurrences are in the same subarray.
        # This means we can find minimal, contiguous segments that must be kept together.
        # If there are `m` such segments, the number of ways to group them is 2^(m-1).
        # We can find `m-1`, which is the number of possible cuts between these segments.

        # Store the last occurrence of each number. This helps determine the minimum
        # range an element's segment must cover.
        last = {num: i for i, num in enumerate(nums)}
            
        cuts = 0
        max_reach = 0
        # Iterate up to the second to last element to find possible cut points.
        # A cut is only possible between elements.
        for i in range(n - 1):
            # The current combined segment must extend at least to the last occurrence
            # of nums[i].
            max_reach = max(max_reach, last[nums[i]])
            
            # If the farthest reach of all numbers in the prefix nums[0..i]
            # is `i` itself, it means this prefix is self-contained.
            # No number in nums[0..i] appears in nums[i+1...n-1].
            # Thus, we can make a valid cut here.
            if max_reach == i:
                cuts += 1
                
        # The number of good partitions is 2^cuts. Each cut point represents a binary
        # choice: to partition there or not.
        return pow(2, cuts, MOD)