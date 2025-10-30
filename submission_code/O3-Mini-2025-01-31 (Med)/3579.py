from typing import List
from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        # Convert each number to its binary representation, skipping the '0b' prefix.
        bin_strs = [bin(num)[2:] for num in nums]
        
        max_value = 0
        # Generate all possible orders (permutations) of the binary strings.
        for perm in permutations(bin_strs):
            # Concatenate the binary strings in the current permutation order.
            combined = "".join(perm)
            # Convert the combined binary string to an integer.
            current_value = int(combined, 2)
            # Update max_value if necessary.
            max_value = max(max_value, current_value)
            
        return max_value

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Test Example 1
    print(sol.maxGoodNumber([1, 2, 3]))  # Expected Output: 30
    # Test Example 2
    print(sol.maxGoodNumber([2, 8, 16]))  # Expected Output: 1296