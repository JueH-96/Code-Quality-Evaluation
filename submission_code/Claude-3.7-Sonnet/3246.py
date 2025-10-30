class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # Count even numbers
        count_even = 0
        for num in nums:
            if num % 2 == 0:  # Check if the number is even
                count_even += 1
                if count_even >= 2:  # Early return if we've found two
                    return True
        return False