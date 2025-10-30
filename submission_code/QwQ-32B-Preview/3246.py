class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # Count the number of even numbers in the array
        even_count = 0
        for num in nums:
            if num % 2 == 0:
                even_count += 1
                # Early exit if we have at least two even numbers
                if even_count >= 2:
                    return True
        # If less than two even numbers, return False
        return False