class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        count = 0
        q = deque([(nums, 0)])
        visited = {tuple(nums)}

        while q:
            curr_nums, curr_ops = q.popleft()
            
            if any(sum(subset) == target for i in range(len(curr_nums) + 1) for subset in combinations(curr_nums, i)):
                return curr_ops

            for i, num in enumerate(curr_nums):
                if num > 1:
                    next_nums = curr_nums[:i] + curr_nums[i+1:] + [num // 2, num // 2]
                    next_nums_tuple = tuple(next_nums)
                    if next_nums_tuple not in visited:
                        visited.add(next_nums_tuple)
                        q.append((next_nums, curr_ops + 1))

        return -1
from collections import deque
from itertools import combinations