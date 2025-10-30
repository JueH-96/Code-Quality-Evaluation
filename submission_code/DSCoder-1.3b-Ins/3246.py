class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        count = 0
        for num in nums:
            while num % 2 == 0:
                num /= 2
            while num % 5 == 0:
                num /= 5
            count += num - int(num)
        return count > 0