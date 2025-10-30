class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        count_zeros = sum(1 for num in nums if num == 0)
        if count_zeros == 0:
            return 0

        q = [(nums, 0)]
        visited = {tuple(nums)}
        while q:
            curr_nums, steps = q.pop(0)
            for i in range(n - 2):
                next_nums = curr_nums[:]
                for j in range(i, i + 3):
                    next_nums[j] = 1 - next_nums[j]
                if all(num == 1 for num in next_nums):
                    return steps + 1
                if tuple(next_nums) not in visited:
                    visited.add(tuple(next_nums))
                    q.append((next_nums, steps + 1))
        return -1