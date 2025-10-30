class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict
        
        # Count occurrences of each point
        point_count = defaultdict(int)
        for x, y in coordinates:
            point_count[(x, y)] += 1
        
        result = 0
        
        # For each unique point
        for (x, y), count in point_count.items():
            # For each possible split of k into a + b
            for a in range(k + 1):
                b = k - a
                target_x = x ^ a
                target_y = y ^ b
                
                if (target_x, target_y) in point_count:
                    result += count * point_count[(target_x, target_y)]
        
        # Adjust for self-pairs and double counting
        if k == 0:
            # For k = 0, we counted count^2 for each point, but we want count * (count - 1)
            for count in point_count.values():
                result -= count
        
        # Divide by 2 to account for double counting
        result //= 2
        
        return result