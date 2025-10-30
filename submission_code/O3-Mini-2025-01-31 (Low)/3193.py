from typing import List

class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        max_xor = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):  # using j starting at i to avoid duplicate checks, since (x,y) and (y,x) yield same XOR and condition symmetric.
                x, y = nums[i], nums[j]
                # Check if the pair is strong: |x-y| <= min(x,y)
                if abs(x - y) <= min(x, y):
                    # Compute XOR for both (x,y) and include it
                    current_xor = x ^ y
                    if current_xor > max_xor:
                        max_xor = current_xor
        return max_xor

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maximumStrongPairXor([1,2,3,4,5]))  # Expected 7
    print(sol.maximumStrongPairXor([10,100]))       # Expected 0
    print(sol.maximumStrongPairXor([5,6,25,30]))    # Expected 7