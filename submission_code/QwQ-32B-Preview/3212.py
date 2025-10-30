class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Build a dictionary to store the last occurrence index of each number
        last = {}
        for idx, num in enumerate(nums):
            last[num] = idx
        
        # Iterate through the array to find the number of groups
        n = len(nums)
        current_end = -1
        group_count = 0
        for i in range(n):
            if i > current_end:
                # Start a new group
                group_count += 1
                current_end = last[nums[i]]
            else:
                # Extend the current group if necessary
                current_end = max(current_end, last[nums[i]])
        
        # Calculate 2^(group_count - 1) mod MOD
        return pow(2, group_count - 1, MOD)