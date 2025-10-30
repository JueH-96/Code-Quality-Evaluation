class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        count = 0
        i = 0
        while nums[i] < k:
            count += 1
            i += 1
            if i == len(nums):
                return -1
        return count