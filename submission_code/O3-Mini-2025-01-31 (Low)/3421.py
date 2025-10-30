from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        # Using remainders mod 24
        freq = [0] * 24
        for h in hours:
            rem = h % 24
            freq[rem] += 1
        
        total_pairs = 0
        # r = 0 pairs
        total_pairs += freq[0] * (freq[0] - 1) // 2
        
        # For remainders r from 1 to 11, complement is (24 - r)
        for r in range(1, 12):
            complement = 24 - r
            total_pairs += freq[r] * freq[complement]
        
        # Check if 24/2 = 12 remainder, pairs among themselves
        total_pairs += freq[12] * (freq[12] - 1) // 2
        
        return total_pairs

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.countCompleteDayPairs([12, 12, 30, 24, 24]))  # Expected 2
    print(sol.countCompleteDayPairs([72, 48, 24, 3]))         # Expected 3