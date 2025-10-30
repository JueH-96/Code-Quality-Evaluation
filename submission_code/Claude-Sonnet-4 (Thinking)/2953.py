class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict
        
        # Count occurrences of each coordinate
        count_map = defaultdict(int)
        for x, y in coordinates:
            count_map[(x, y)] += 1
        
        pairs = 0
        
        # For each unique coordinate, find all coordinates at distance k
        for (x1, y1), count1 in count_map.items():
            for dx in range(k + 1):
                dy = k - dx
                x2 = x1 ^ dx
                y2 = y1 ^ dy
                
                if (x2, y2) in count_map:
                    count2 = count_map[(x2, y2)]
                    if (x1, y1) == (x2, y2):
                        # Same point, choose 2 from count1
                        pairs += count1 * (count1 - 1) // 2
                    elif (x1, y1) < (x2, y2):  # To avoid double counting
                        pairs += count1 * count2
        
        return pairs