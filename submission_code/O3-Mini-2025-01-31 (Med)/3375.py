from typing import List
import math

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Precompute non-empty subsets of coins along with their lcm and inclusion sign.
        # For a given candidate x, using inclusion-exclusion,
        # count(x) = sum_over_subsets [sign * (x // lcm(subset))]
        # where sign = +1 if subset length is odd, -1 if even.
        n = len(coins)
        subsets = []
        
        # Helper function to compute lcm of two numbers.
        def lcm(a: int, b: int) -> int:
            return a // math.gcd(a, b) * b
        
        # Iterate over all possible non-empty subsets (using bitmask).
        for mask in range(1, 1 << n):
            current_lcm = 1
            bit_count = 0
            for i in range(n):
                if mask & (1 << i):
                    bit_count += 1
                    current_lcm = lcm(current_lcm, coins[i])
            sign = 1 if bit_count % 2 == 1 else -1
            subsets.append((current_lcm, sign))
        
        # Function to count how many numbers <= x are multiples of at least one coin.
        def count_multiples(x: int) -> int:
            total = 0
            for l, sign in subsets:
                total += sign * (x // l)
            return total
        
        # Binary search: kth smallest number that can be made is the smallest x such that count_multiples(x) >= k.
        lo, hi = 1, min(coins) * k  # hi bound: worst-case when only the smallest coin determines the kth multiple.
        while lo < hi:
            mid = (lo + hi) // 2
            if count_multiples(mid) < k:
                lo = mid + 1
            else:
                hi = mid
        return lo

# You can run the following tests to check your solution.
if __name__ == "__main__":
    sol = Solution()
    print(sol.findKthSmallest([3, 6, 9], 3))  # Expected output: 9
    print(sol.findKthSmallest([5, 2], 7))     # Expected output: 12