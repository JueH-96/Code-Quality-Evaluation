from typing import List

class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # A number's bitwise OR will have at least one trailing zero
        # exactly when its least significant bit is 0, i.e., the OR is even.
        # The OR of a set of numbers is even if and only if all those numbers
        # are even (their LSBs are 0). So we just need at least two evens.
        even_count = sum(1 for x in nums if x % 2 == 0)
        return even_count >= 2


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.hasTrailingZeros([1,2,3,4,5]))  # True
    print(sol.hasTrailingZeros([2,4,8,16]))   # True
    print(sol.hasTrailingZeros([1,3,5,7,9]))  # False