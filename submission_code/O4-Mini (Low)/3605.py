from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for p in nums:
            res = -1
            # Try all possible a from 0 up to p (inclusive)
            for a in range(p + 1):
                if (a | (a + 1)) == p:
                    res = a
                    break
            ans.append(res)
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minBitwiseArray([2, 3, 5, 7]))   # [-1, 1, 4, 3]
    print(sol.minBitwiseArray([11, 13, 31]))   # [9, 12, 15]