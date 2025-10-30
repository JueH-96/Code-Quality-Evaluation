from typing import List

class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # To have trailing zeros in the binary representation, the number must be even.
        # When performing OR, the least significant bit remains 0 only if every number in the OR is even.
        # Therefore, we just need to check if there are at least 2 even numbers in the list.
        even_count = sum(1 for num in nums if num % 2 == 0)
        return even_count >= 2

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.hasTrailingZeros([1,2,3,4,5]))  # Expected: True
    print(sol.hasTrailingZeros([2,4,8,16]))   # Expected: True
    print(sol.hasTrailingZeros([1,3,5,7,9]))  # Expected: False