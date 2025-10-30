class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict
        
        count = 0
        n = len(coordinates)
        freq = defaultdict(int)
        
        for i in range(n):
            x, y = coordinates[i]
            # We need to find pairs (i, j) such that (x_i XOR x_j) + (y_i XOR y_j) = k
            # Rearranging gives us: (x_i XOR x_j) = k - (y_i XOR y_j)
            for j in range(i):
                x_j, y_j = coordinates[j]
                distance = (x ^ x_j) + (y ^ y_j)
                if distance == k:
                    count += 1
        
        return count