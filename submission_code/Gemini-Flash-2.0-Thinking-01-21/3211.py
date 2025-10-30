class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        non_decreasing_sums = []
        current_sum = 0
        for num in nums:
            current_sum += num
            if not non_decreasing_sums or current_sum >= non_decreasing_sums[-1]:
                non_decreasing_sums.append(current_sum)
                current_sum = 0
        return len(non_decreasing_sums)