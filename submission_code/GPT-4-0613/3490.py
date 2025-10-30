class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        odd, even = 0, 0
        for num in nums:
            if num % 2 == 0:
                even = max(even, odd + 1)
            else:
                odd = max(odd, even + 1)
        return max(odd, even)