from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Check vertical cuts and horizontal cuts
        return self.check(n, rectangles, 0) or self.check(n, rectangles, 1)
    
    def check(self, n, rectangles, coord):
        # coord: 0 for x (vertical cuts), 1 for y (horizontal cuts)
        # Sort rectangles based on end coordinate
        sorted_rects = sorted(rectangles, key=lambda x: x[coord + 2])
        m = len(sorted_rects)
        if m < 3:
            return False
        
        # Precompute max_end array
        max_end = [0] * m
        max_end[0] = sorted_rects[0][coord + 2]
        for i in range(1, m):
            max_end[i] = max(max_end[i-1], sorted_rects[i][coord + 2])
        
        # Precompute min_start_suffix array
        min_start_suffix = [0] * m
        min_start_suffix[-1] = sorted_rects[-1][coord]
        for i in range(m-2, -1, -1):
            min_start_suffix[i] = min(sorted_rects[i][coord], min_start_suffix[i+1])
        
        # Precompute split_points and possible_split array
        split_points = [False] * m
        for i in range(m-1):
            split_points[i] = (sorted_rects[i][coord + 2] < min_start_suffix[i+1])
        split_points[-1] = False  # No rectangles after the last one
        
        possible_split = [False] * m
        possible_split[-1] = split_points[-1]
        for i in range(m-2, -1, -1):
            possible_split[i] = split_points[i] or possible_split[i+1]
        
        # Now, iterate over each possible i
        for i in range(m-2):
            a = max_end[i]
            # Find j using binary search: first index >i where start > a
            low = i + 1
            high = m - 1
            j = -1
            while low <= high:
                mid = (low + high) // 2
                if sorted_rects[mid][coord] > a:
                    j = mid
                    high = mid - 1
                else:
                    low = mid + 1
            if j == -1:
                continue
            if j >= m:
                continue
            # Check if there's a possible split point starting at j
            if j < m-1 and possible_split[j]:
                return True
        return False