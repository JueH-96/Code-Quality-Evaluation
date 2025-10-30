class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Find first and last occurrence of each number
        first = {}
        last = {}
        
        for i, num in enumerate(nums):
            if num not in first:
                first[num] = i
            last[num] = i
        
        # Find the rightmost position we need to reach for each starting position
        max_end = [0] * n
        for i in range(n):
            max_end[i] = last[nums[i]]
        
        # Find positions where we can make a cut
        # We can cut after position i if all numbers that start at or before i
        # also end at or before i
        cuts = 0
        current_max_end = 0
        
        for i in range(n - 1):  # We don't consider cutting after the last element
            current_max_end = max(current_max_end, max_end[i])
            if current_max_end == i:
                # We can make a cut after position i
                cuts += 1
        
        # Number of ways = 2^cuts
        return pow(2, cuts, MOD)