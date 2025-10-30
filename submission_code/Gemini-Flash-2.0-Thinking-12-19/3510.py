class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        indexed_max_heights = []
        for i in range(n):
            indexed_max_heights.append((maximumHeight[i], i))
        
        indexed_max_heights.sort(key=lambda x: x[0], reverse=True)
        
        assigned_heights = [0] * n
        used_heights = set()
        
        for max_h, index in indexed_max_heights:
            assigned_height = -1
            for h in range(max_h, 0, -1):
                if h not in used_heights:
                    assigned_height = h
                    break
            if assigned_height == -1:
                return -1
            assigned_heights[index] = assigned_height
            used_heights.add(assigned_height)
            
        return sum(assigned_heights)