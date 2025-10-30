from typing import List
from collections import Counter

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        count = Counter(nums)
        result = 0
        for num, freq in count.items():
            if freq == 2:
                result ^= num
        return result

# Example test cases:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.duplicateNumbersXOR([1,2,1,3]))  # Expected output: 1
    # Example 2:
    print(sol.duplicateNumbersXOR([1,2,3]))    # Expected output: 0
    # Example 3:
    print(sol.duplicateNumbersXOR([1,2,2,1]))  # Expected output: 3