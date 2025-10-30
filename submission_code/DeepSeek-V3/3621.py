class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        if all(num == k for num in nums):
            return 0
        if any(num < k for num in nums):
            return -1
        operations = 0
        while True:
            current_max = max(nums)
            if current_max == k:
                break
            h = current_max
            if not all(num <= h or num == current_max for num in nums):
                return -1
            for i in range(len(nums)):
                if nums[i] > h:
                    nums[i] = h
            operations += 1
        return operations