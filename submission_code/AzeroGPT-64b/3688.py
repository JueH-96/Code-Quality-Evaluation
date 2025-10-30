class Solution:
    def maxSubarraySumAfterDeletion(self, nums: List[int]) -> int:
        all_sum = sum(nums)
        count = Counter(nums)
        max_sum = nums_sum = nums_count = 0

        for num, cnt in sorted(count.items()):
            nums_sum += num * cnt
            nums_count += cnt
            if nums_count == len(nums):
                max_sum = max(max_sum, nums_sum - all_sum + num * cnt)
            else:
                max_sum = max(max_sum, nums_sum)

        return max_sum