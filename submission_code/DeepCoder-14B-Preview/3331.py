class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        sorted_nums = sorted(nums)
        count = 0
        for num in sorted_nums:
            if num < k:
                count += 1
            else:
                break
        return count