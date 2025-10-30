class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # Check if there are at least two numbers with trailing zeros
        count = 0
        for num in nums:
            if num & 1 == 0:  # Check if the number has a trailing zero
                count += 1
            if count >= 2:
                return True
        return False