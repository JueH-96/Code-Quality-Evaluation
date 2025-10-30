from typing import List

class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # To have at least one trailing zero in the OR result,
        # the result must be even – which means its least significant bit is 0.
        # In a bitwise OR, if any number has a 1 in the least significant bit (i.e., is odd),
        # then the final result will also be odd.
        # Thus, we need to select two or more numbers that are all even.
        
        even_count = sum(1 for num in nums if num % 2 == 0)
        return even_count >= 2

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.hasTrailingZeros([1,2,3,4,5]))  # Expected output: True
    print(sol.hasTrailingZeros([2,4,8,16]))   # Expected output: True
    print(sol.hasTrailingZeros([1,3,5,7,9]))  # Expected output: False