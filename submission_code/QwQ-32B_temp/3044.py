class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        last_occurrence = {}
        n = len(nums)
        for i in range(n):
            last_occurrence[nums[i]] = i  # Store the last occurrence index of each number
        max_steps = 0
        for x in range(1, k + 1):
            i = last_occurrence[x]
            steps = n - i
            if steps > max_steps:
                max_steps = steps
        return max_steps