from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        # For each prime number p in nums, we need to find the smallest integer x
        # (with x >= 0 and x < p) such that x | (x+1) == p.
        #
        # Observation:
        # - For any even x, note that no carry occurs when adding 1.
        #   In that case x has a trailing 0 and x+1 has a trailing 1,
        #   so x | (x+1) is simply x with the least-significant bit forced to 1,
        #   i.e. x | (x+1) == x+1.
        #   Hence one possible solution for odd p would be x = p-1.
        # - However, often when x is odd, due to a carry in x+1,
        #   the OR can “fill in” bits in a way that yields p even when x < p-1.
        #
        # In our constrained problem (p is prime and 2 <= p <= 1000),
        # a simple brute force check x from 0 to p-1 (p candidates) is efficient.
        #
        # Special case: p = 2. Testing:
        # x = 0:  0 | 1 = 1  (≠ 2)
        # x = 1:  1 | 2 = 3  (≠ 2)
        # So in that case no valid x exists, and we output -1.
        #
        for p in nums:
            if p == 2:
                ans.append(-1)
            else:
                candidate = -1
                for x in range(p):
                    if (x | (x + 1)) == p:
                        candidate = x
                        break
                ans.append(candidate)
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    nums1 = [2, 3, 5, 7]
    print(sol.minBitwiseArray(nums1))  # Expected output: [-1, 1, 4, 3]
    
    # Example 2:
    nums2 = [11, 13, 31]
    print(sol.minBitwiseArray(nums2))  # Expected output: [9, 12, 15]