from itertools import combinations

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        def strictly_increasing(arr):
            return all(x < y for x, y in zip(arr, arr[1:]))
        
        count = 0
        for i in range(1, n + 1):
            for combination in combinations(range(n), i):
                new_arr = [nums[j] for j in range(n) if j not in combination]
                if strictly_increasing(new_arr):
                    count += 1
        return count