from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        if self.check_horizontal(rectangles) or self.check_vertical(rectangles):
            return True
        return False
    
    def check_horizontal(self, rects: List[List[int]]) -> bool:
        rects_sorted = sorted(rects, key=lambda x: x[3])
        m = len(rects_sorted)
        min_sy_suffix = [0] * m
        cur_min = float('inf')
        for i in range(m-1, -1, -1):
            cur_min = min(cur_min, rects_sorted[i][1])
            min_sy_suffix[i] = cur_min
        
        valid_k = [False] * m
        for k in range(m-1):
            if rects_sorted[k][3] <= min_sy_suffix[k+1]:
                valid_k[k] = True
        
        suffix_valid = [False] * (m + 1)
        suffix_valid[m] = False
        for i in range(m-1, -1, -1):
            if i == m-1:
                suffix_valid[i] = False
            else:
                suffix_valid[i] = valid_k[i] or suffix_valid[i+1]
        
        for i in range(1, m-1):
            y1 = rects_sorted[i-1][3]
            if min_sy_suffix[i] < y1:
                continue
            if suffix_valid[i]:
                return True
        return False

    def check_vertical(self, rects: List[List[int]]) -> bool:
        rects_sorted = sorted(rects, key=lambda x: x[2])
        m = len(rects_sorted)
        min_sx_suffix = [0] * m
        cur_min = float('inf')
        for i in range(m-1, -1, -1):
            cur_min = min(cur_min, rects_sorted[i][0])
            min_sx_suffix[i] = cur_min
        
        valid_k = [False] * m
        for k in range(m-1):
            if rects_sorted[k][2] <= min_sx_suffix[k+1]:
                valid_k[k] = True
        
        suffix_valid = [False] * (m + 1)
        suffix_valid[m] = False
        for i in range(m-1, -1, -1):
            if i == m-1:
                suffix_valid[i] = False
            else:
                suffix_valid[i] = valid_k[i] or suffix_valid[i+1]
        
        for i in range(1, m-1):
            x1 = rects_sorted[i-1][2]
            if min_sx_suffix[i] < x1:
                continue
            if suffix_valid[i]:
                return True
        return False