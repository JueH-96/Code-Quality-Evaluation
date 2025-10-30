class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        dp_sums = []
        current_sum = 0
        for num in nums:
            current_sum += num
            if not dp_sums or current_sum >= dp_sums[-1]:
                dp_sums.append(current_sum)
                current_sum = 0
        return len(dp_sums)