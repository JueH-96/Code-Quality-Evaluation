class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        if self.check_horizontal(rectangles):
            return True
        if self.check_vertical(rectangles):
            return True
        return False

    def check_horizontal(self, rectangles):
        sorted_rect = sorted(rectangles, key=lambda r: r[3])
        n_rect = len(sorted_rect)
        if n_rect < 3:
            return False
        
        suf_min = [0] * n_rect
        suf_min[-1] = sorted_rect[-1][1]
        for i in range(n_rect - 2, -1, -1):
            suf_min[i] = min(sorted_rect[i][1], suf_min[i + 1])
        
        last_occurrence = {}
        for i, r in enumerate(sorted_rect):
            last_occurrence[r[3]] = i
        
        distinct_a = sorted(last_occurrence.keys())
        for a_val in distinct_a:
            i = last_occurrence[a_val]
            if i >= n_rect - 2:
                continue
            if suf_min[i + 1] < a_val:
                continue
            remaining = sorted_rect[i + 1:]
            rem_y = [(r[1], r[3]) for r in remaining]
            rem_y.sort(key=lambda x: x[0])
            max_y2 = 0
            for j in range(len(rem_y) - 1):
                max_y2 = max(max_y2, rem_y[j][1])
                if max_y2 <= rem_y[j + 1][0]:
                    return True
        return False

    def check_vertical(self, rectangles):
        sorted_rect = sorted(rectangles, key=lambda r: r[2])
        n_rect = len(sorted_rect)
        if n_rect < 3:
            return False
        
        suf_min = [0] * n_rect
        suf_min[-1] = sorted_rect[-1][0]
        for i in range(n_rect - 2, -1, -1):
            suf_min[i] = min(sorted_rect[i][0], suf_min[i + 1])
        
        last_occurrence = {}
        for i, r in enumerate(sorted_rect):
            last_occurrence[r[2]] = i
        
        distinct_a = sorted(last_occurrence.keys())
        for a_val in distinct_a:
            i = last_occurrence[a_val]
            if i >= n_rect - 2:
                continue
            if suf_min[i + 1] < a_val:
                continue
            remaining = sorted_rect[i + 1:]
            rem_x = [(r[0], r[2]) for r in remaining]
            rem_x.sort(key=lambda x: x[0])
            max_x2 = 0
            for j in range(len(rem_x) - 1):
                max_x2 = max(max_x2, rem_x[j][1])
                if max_x2 <= rem_x[j + 1][0]:
                    return True
        return False