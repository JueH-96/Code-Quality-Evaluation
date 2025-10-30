from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        
        # For every value, record its last occurrence in the array.
        last_occurrence = {}
        for i, num in enumerate(nums):
            last_occurrence[num] = i

        # The idea is that a partition is valid if each distinct number appears in exactly one partition.
        # This is equivalent to saying that whenever we decide to make a partition cut at some index i
        # (thus ending a subarray), all numbers in that subarray must have their last occurrence <= i.
        # We can scan the array while maintaining the furthest (maximum) last occurrence encountered in the current block.
        # When the current index equals that maximum, it is safe to place a cut (i.e. end a block).
        # Notice that the entire array splits into a certain number of "safe blocks".
        # Once we have these safe blocks, any valid partition
        # is simply a way to combine consecutive safe blocks into larger segments.
        # If there are m safe blocks, there are exactly 2^(m-1) ways to partition them (where we decide on the boundaries between safe blocks).
        
        safe_block_count = 0
        current_max = 0
        
        for i, num in enumerate(nums):
            current_max = max(current_max, last_occurrence[num])
            if i == current_max:
                safe_block_count += 1
        
        # For m safe blocks, ways = 2^(m-1).
        return pow(2, safe_block_count - 1, mod)


# The following is a standalone wrapper to enable testing the code.
if __name__ == '__main__':
    import sys

    # If running as a script, read from standard input.
    import sys
    data = sys.stdin.read().split()
    if data:
        # the input format is assumed to be:
        # first an integer n (the length of the array) followed by n integers.
        n = int(data[0])
        nums = list(map(int, data[1:1+n]))
        sol = Solution()
        result = sol.numberOfGoodPartitions(nums)
        sys.stdout.write(str(result))
    else:
        # If no input is provided, run some example test cases.
        sol = Solution()
        print(sol.numberOfGoodPartitions([1, 2, 3, 4]))   # Expected output: 8
        print(sol.numberOfGoodPartitions([1, 1, 1, 1]))   # Expected output: 1
        print(sol.numberOfGoodPartitions([1, 2, 1, 3]))   # Expected output: 2