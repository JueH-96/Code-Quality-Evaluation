class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        indexed_max_heights = []
        for i in range(n):
            indexed_max_heights.append((maximumHeight[i], i))
        indexed_max_heights.sort(key=lambda x: x[0], reverse=True)
        assigned_heights = [-1] * n
        used_heights = set()
        for max_height, index in indexed_max_heights:
            assigned = False
            for h in range(max_height, 0, -1):
                if h not in used_heights:
                    assigned_heights[index] = h
                    used_heights.add(h)
                    assigned = True
                    break
            if not assigned:
                return -1
        return sum(assigned_heights)