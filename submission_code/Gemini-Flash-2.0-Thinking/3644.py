class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        min_positive_sum = float('inf')
        found = False

        for length in range(l, r + 1):
            if length > n:
                continue

            current_sum = sum(nums[:length])
            if current_sum > 0:
                min_positive_sum = min(min_positive_sum, current_sum)
                found = True

            for i in range(1, n - length + 1):
                current_sum = current_sum - nums[i - 1] + nums[i + length - 1]
                if current_sum > 0:
                    min_positive_sum = min(min_positive_sum, current_sum)
                    found = True

        if found:
            return min_positive_sum
        else:
            return -1