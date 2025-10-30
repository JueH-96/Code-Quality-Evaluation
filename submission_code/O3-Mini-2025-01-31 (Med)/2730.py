from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Precompute left OR prefix array.
        left = [0] * n
        for i in range(1, n):
            left[i] = left[i - 1] | nums[i - 1]
                
        # Precompute right OR suffix array.
        right = [0] * n
        for i in range(n - 2, -1, -1):
            right[i] = right[i + 1] | nums[i + 1]
                
        # We can use the operation on exactly one element (applying all k shifts)
        # and then combine its modified value with the OR of the others.
        max_or = 0
        factor = (1 << k)  # each operation doubles => k times doubling multiplies by 2^k.
        for i in range(n):
            current = left[i] | (nums[i] * factor) | right[i]
            max_or = max(max_or, current)
            
        return max_or

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Test case 1:
    print(sol.maximumOr([12, 9], 1))  # Expected output: 30
    # Test case 2:
    print(sol.maximumOr([8, 1, 2], 2))  # Expected output: 35