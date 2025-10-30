from typing import List
import collections

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        # Build a frequency dictionary for each unique coordinate.
        freq = collections.Counter(tuple(coord) for coord in coordinates)
        
        # Special case: When k == 0, the condition reduces to 
        # (x XOR x) + (y XOR y) = 0, so the two points must be identical.
        if k == 0:
            return sum(count * (count - 1) // 2 for count in freq.values())
        
        # For k > 0, consider all splits of k into two nonnegative integers:
        # Let diff_x = x1 XOR x2 and diff_y = y1 XOR y2.
        # We require diff_x + diff_y = k.
        ans = 0
        items = list(freq.items())  # List of (coordinate, frequency) pairs.
        for diff_x in range(k + 1):
            diff_y = k - diff_x
            # For each coordinate (x, y) with its frequency,
            # the candidate coordinate that yields the required XOR differences is:
            # (x^diff_x, y^diff_y).
            for (x, y), count in items:
                candidate = (x ^ diff_x, y ^ diff_y)
                if candidate in freq:
                    # To avoid double counting unordered pairs, only count when
                    # (x,y) is lexicographically smaller than candidate.
                    if (x, y) < candidate:
                        ans += count * freq[candidate]
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.countPairs([[1,2],[4,2],[1,3],[5,2]], 5))   # Expected output: 2
    # Example 2:
    print(sol.countPairs([[1,3],[1,3],[1,3],[1,3],[1,3]], 0))  # Expected output: 10