class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        max_removals = 0

        for i in range(1 << n):  # Iterate through all possible subsets of queries
            curr_nums = nums.copy()
            removals = 0
            for j in range(n):
                if i & (1 << j):  # Check if the jth query is included in the current subset
                    l, r = queries[j]
                    for k in range(l, r + 1):
                        curr_nums[k] = max(0, curr_nums[k] - 1)
                    removals += 1
            if all(x == 0 for x in curr_nums):  # Check if the current nums array is a zero array
                max_removals = max(max_removals, removals)

        return max_removals if max_removals > 0 else -1