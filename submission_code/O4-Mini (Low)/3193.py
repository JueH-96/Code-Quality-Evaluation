from typing import List

class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        """
        We need to find two numbers x, y in nums (possibly the same)
        such that |x - y| <= min(x, y) and x ^ y is maximized.
        Since nums.length <= 50, we can brute force all pairs.
        """
        ans = 0
        n = len(nums)
        for i in range(n):
            for j in range(n):
                x, y = nums[i], nums[j]
                # Check the "strong pair" condition
                if abs(x - y) <= min(x, y):
                    ans = max(ans, x ^ y)
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maximumStrongPairXor([1,2,3,4,5]))  # Expected 7
    print(sol.maximumStrongPairXor([10,100]))      # Expected 0
    print(sol.maximumStrongPairXor([5,6,25,30]))   # Expected 7