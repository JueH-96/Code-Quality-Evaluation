class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        flipped = [False] * n

        for i in range(n - 2):
            if (nums[i] == 0 and not flipped[i]) or (nums[i] == 1 and flipped[i]):
                operations += 1
                flipped[i] = not flipped[i]
                flipped[i + 1] = not flipped[i + 1]
                flipped[i + 2] = not flipped[i + 2]

        if (nums[n - 2] == 0 and not flipped[n - 2]) or (nums[n - 2] == 1 and flipped[n - 2]):
            return -1
        if (nums[n - 1] == 0 and not flipped[n - 1]) or (nums[n - 1] == 1 and flipped[n - 1]):
            return -1

        return operations