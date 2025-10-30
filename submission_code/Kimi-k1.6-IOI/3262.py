class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        current_sum = 0
        max_peri = -1
        for num in nums:
            if current_sum > num:
                max_peri = max(max_peri, current_sum + num)
            current_sum += num
        return max_peri