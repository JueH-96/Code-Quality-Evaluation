class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import Counter
        
        # Count how many times each (x, y) appears
        freq = Counter((x, y) for x, y in coordinates)
        
        result = 0
        
        # Iterate over all possible (dx, dy) with dx + dy = k
        for dx in range(k+1):
            dy = k - dx
            # Handle the case dx = 0, dy = 0 separately (pairs within the same group)
            if dx == 0 and dy == 0:
                for count in freq.values():
                    # Number of ways to pick 2 out of 'count' identical points
                    result += count * (count - 1) // 2
            else:
                # For non-zero (dx, dy), we can accumulate in a temporary sum
                partial_sum = 0
                for (x, y), c1 in freq.items():
                    c2 = freq.get((x ^ dx, y ^ dy), 0)
                    partial_sum += c1 * c2
                # Each distinct pair is counted twice (once from each side), so halve it
                result += partial_sum // 2
        
        return result