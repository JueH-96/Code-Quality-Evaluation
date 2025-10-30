from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Since in one operation we remove only one occurrence of the smallest element,
        # and if any element in nums is less than k then it has to be removed eventually,
        # the answer is simply the count of elements in nums that are strictly less than k.
        return sum(1 for num in nums if num < k)
        
# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.minOperations([2, 11, 10, 1, 3], 10))  # Expected output: 3
    # Example 2
    print(sol.minOperations([1, 1, 2, 4, 9], 1))       # Expected output: 0
    # Example 3
    print(sol.minOperations([1, 1, 2, 4, 9], 9))       # Expected output: 4