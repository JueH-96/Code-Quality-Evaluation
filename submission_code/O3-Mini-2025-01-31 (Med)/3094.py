from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Count the frequency of each number.
        count = Counter(nums)
        total_operations = 0
        
        # For each distinct element, we want to partition its frequency f
        # into groups of 3 (removing 3 elements) and groups of 2 (removing 2 elements)
        # so that 3*b + 2*a = f and we want to minimize (a + b) (the total number of operations).
        # Notice that since removing 3 elements in one op is "more efficient", we want to use as many
        # operations of 3 as possible subject to the condition that the remainder (f - 3*b) is divisible by 2.
        #
        # We can derive a condition:
        # Let b be the maximum number of triple removals such that b <= f//3 and
        # (f - 3*b) is even. Equivalently, note that:
        #   f - 3*b ≡ f - b  (mod 2) [since 3 ≡ 1 (mod 2)]
        # So we need: f - b ≡ 0 (mod 2) or b ≡ f (mod 2).
        # Therefore, let candidate b = f//3. If candidate % 2 != f % 2 then we decrement candidate by 1.
        # If candidate is negative then there is no solution.
        #
        # Then the remaining elements will be removed in pairs:
        #   a = (f - 3*b) / 2.
        # The number of operations for frequency f is then (a + b).
        
        for f in count.values():
            b = f // 3
            if b % 2 != (f % 2):
                b -= 1
            if b < 0:
                return -1
            remaining = f - 3 * b
            if remaining % 2 != 0:
                return -1
            a = remaining // 2
            total_operations += (a + b)
        
        return total_operations

# For local testing
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.minOperations([2,3,3,2,2,4,2,3,4]))  # Expected output: 4
    # Example 2:
    print(sol.minOperations([2,1,2,2,3,3]))         # Expected output: -1