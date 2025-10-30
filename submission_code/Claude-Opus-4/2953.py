class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict
        
        # Count frequency of each coordinate
        coord_count = defaultdict(int)
        for x, y in coordinates:
            coord_count[(x, y)] += 1
        
        result = 0
        
        # For each unique coordinate
        for (x1, y1), count1 in coord_count.items():
            # Try all possible ways to split k between x and y XOR values
            for i in range(k + 1):
                # We need x1 XOR x2 = i and y1 XOR y2 = k - i
                # So x2 = x1 XOR i and y2 = y1 XOR (k - i)
                x2 = x1 ^ i
                y2 = y1 ^ (k - i)
                
                if (x2, y2) in coord_count:
                    count2 = coord_count[(x2, y2)]
                    
                    if (x1, y1) == (x2, y2):
                        # Same coordinate - choose 2 from count1
                        result += count1 * (count1 - 1) // 2
                    elif (x1, y1) < (x2, y2):
                        # Different coordinates - multiply counts
                        # Only count once by ordering
                        result += count1 * count2
        
        return result