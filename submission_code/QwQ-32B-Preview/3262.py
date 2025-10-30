class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums_sorted = sorted(nums)
        total_sum = sum(nums_sorted)
        n = len(nums_sorted)
        k = n
        while k >= 3:
            if total_sum - nums_sorted[k-1] > nums_sorted[k-1]:
                return total_sum
            else:
                total_sum -= nums_sorted[k-1]
                k -= 1
        return -1