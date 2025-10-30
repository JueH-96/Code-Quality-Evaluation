import bisect

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def check_cuts(dim):
            start_idx = dim
            end_idx = dim + 2
            coords = []
            for rect in rectangles:
                coords.append(rect[start_idx])
                coords.append(rect[end_idx])
            coords.sort()
            
            sorted_rects = sorted(rectangles, key=lambda x: x[start_idx])
            starts_sorted = [rect[start_idx] for rect in sorted_rects]
            ends = [rect[end_idx] for rect in rectangles]
            ends_sorted = sorted(ends)
            
            m = len(sorted_rects)
            if m == 0:
                return False
            
            suffix_min = [0] * m
            suffix_min[-1] = sorted_rects[-1][end_idx]
            for i in range(m-2, -1, -1):
                suffix_min[i] = min(sorted_rects[i][end_idx], suffix_min[i+1])
            
            for i in range(len(coords)-1):
                prev = coords[i]
                curr = coords[i+1]
                if prev >= curr:
                    continue
                if ends_sorted[0] > prev:
                    continue
                
                pos = bisect.bisect_left(starts_sorted, curr)
                if pos >= len(starts_sorted):
                    continue
                
                idx = bisect.bisect_left(starts_sorted, prev)
                if idx >= m:
                    continue
                if suffix_min[idx] <= curr:
                    return True
            return False
        
        return check_cuts(0) or check_cuts(1)