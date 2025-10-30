from typing import List

class Solution:
    class SegmentTree:
        def __init__(self, data):
            self.n = len(data)
            self.size = 1
            while self.size < self.n:
                self.size <<= 1
            self.tree = [-float('inf')] * (2 * self.size)
            for i in range(self.n):
                self.tree[self.size + i] = data[i]
            for i in range(self.size - 1, 0, -1):
                self.tree[i] = max(self.tree[2 * i], self.tree[2 * i + 1])
        
        def query(self, l, r):
            res = -float('inf')
            l += self.size
            r += self.size
            while l <= r:
                if l % 2 == 1:
                    res = max(res, self.tree[l])
                    l += 1
                if r % 2 == 0:
                    res = max(res, self.tree[r])
                    r -= 1
                l >>= 1
                r >>= 1
            return res

    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        if not rectangles:
            return False

        # Check horizontal case
        sorted_horizontal = sorted(rectangles, key=lambda r: r[3])
        suffix_min_start = [0] * len(sorted_horizontal)
        suffix_min_start[-1] = sorted_horizontal[-1][1]
        for i in range(len(sorted_horizontal) - 2, -1, -1):
            suffix_min_start[i] = min(sorted_horizontal[i][1], suffix_min_start[i + 1])
        
        data_h = [r[1] for r in sorted_horizontal]
        st_h = self.SegmentTree(data_h)
        
        for i in range(len(sorted_horizontal)):
            a_candidate = sorted_horizontal[i][3]
            remaining_start = i + 1
            if remaining_start >= len(sorted_horizontal):
                continue
            b_candidate = suffix_min_start[remaining_start]
            if a_candidate > b_candidate:
                continue
            
            low, high = remaining_start, len(sorted_horizontal) - 1
            j = remaining_start - 1
            while low <= high:
                mid = (low + high) // 2
                if sorted_horizontal[mid][3] <= b_candidate:
                    j = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if j < remaining_start:
                continue
            
            max_start = st_h.query(remaining_start, j)
            if max_start >= a_candidate:
                return True

        # Check vertical case
        sorted_vertical = sorted(rectangles, key=lambda r: r[2])
        suffix_min_start_x = [0] * len(sorted_vertical)
        suffix_min_start_x[-1] = sorted_vertical[-1][0]
        for i in range(len(sorted_vertical) - 2, -1, -1):
            suffix_min_start_x[i] = min(sorted_vertical[i][0], suffix_min_start_x[i + 1])
        
        data_v = [r[0] for r in sorted_vertical]
        st_v = self.SegmentTree(data_v)
        
        for i in range(len(sorted_vertical)):
            a_candidate = sorted_vertical[i][2]
            remaining_start = i + 1
            if remaining_start >= len(sorted_vertical):
                continue
            b_candidate = suffix_min_start_x[remaining_start]
            if a_candidate > b_candidate:
                continue
            
            low, high = remaining_start, len(sorted_vertical) - 1
            j = remaining_start - 1
            while low <= high:
                mid = (low + high) // 2
                if sorted_vertical[mid][2] <= b_candidate:
                    j = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if j < remaining_start:
                continue
            
            max_start_x = st_v.query(remaining_start, j)
            if max_start_x >= a_candidate:
                return True

        return False