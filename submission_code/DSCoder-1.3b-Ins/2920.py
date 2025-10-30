class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        min_nums = min(nums)
        max_nums = max(nums)
        if min_nums == max_nums:
            return 1
        else:
            min_seconds = (max_nums - min_nums) // (n - 1)
            if (max_nums - min_nums) % (n - 1) != 0:
                min_seconds += 1
            return min_seconds