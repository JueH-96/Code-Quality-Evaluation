class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        count_even = 0
        for num in nums:
            if num % 2 == 0:
                count_even += 1
                if count_even >= 2:
                    return True
        return False