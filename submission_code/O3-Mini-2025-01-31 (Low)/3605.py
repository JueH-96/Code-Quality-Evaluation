from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        def count_trailing_ones(n: int) -> int:
            count = 0
            while n & 1:
                count += 1
                n >>= 1
            return count
        
        ans = []
        for num in nums:
            # If num is even, no such x exists
            if num % 2 == 0:
                ans.append(-1)
            else:
                # Count trailing ones in num (binary representation)
                # Since num is odd, there is at least one trailing 1.
                L = count_trailing_ones(num)
                # Minimal x is num - 2^(L-1)
                candidate = num - (1 << (L - 1))
                ans.append(candidate)
        return ans

# For local testing:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minBitwiseArray([2,3,5,7]))   # Expected output: [-1,1,4,3]
    print(sol.minBitwiseArray([11,13,31]))    # Expected output: [9,12,15]