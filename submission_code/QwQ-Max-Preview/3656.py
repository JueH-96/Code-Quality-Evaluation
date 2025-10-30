class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        k = 0
        while True:
            start = 3 * k
            if start >= len(nums):
                return k
            sub = nums[start:]
            if len(set(sub)) == len(sub):
                return k
            k += 1