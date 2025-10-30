from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        res = []
        for m in nums:
            candidate = -1
            # m is at most 1000 so iterating from 0 to m is efficient.
            for x in range(m+1):
                # Check if x OR (x+1) equals m.
                if (x | (x + 1)) == m:
                    candidate = x
                    break
            res.append(candidate)
        return res

# Sample test cases
if __name__ == '__main__':
    sol = Solution()
    print(sol.minBitwiseArray([2,3,5,7]))   # Expected output: [-1, 1, 4, 3]
    print(sol.minBitwiseArray([11,13,31]))    # Expected output: [9, 12, 15]