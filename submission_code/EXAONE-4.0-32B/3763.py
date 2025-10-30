from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        total_area = 0
        events_dict = {}
        for sq in squares:
            x, y, l_val = sq
            total_area += l_val * l_val
            if y not in events_dict:
                events_dict[y] = []
            events_dict[y].append(('start', l_val, y))
            end_y = y + l_val
            if end_y not in events_dict:
                events_dict[end_y] = []
            events_dict[end_y].append(('end', l_val, y))
        
        half_area = total_area / 2.0
        
        ys = sorted(events_dict.keys())
        active_sum_l = 0.0
        active_sum_ly = 0.0
        fixed_area = 0.0
        prev_y = None
        
        for y_val in ys:
            if prev_y is not None:
                F_prev = fixed_area + active_sum_l * prev_y - active_sum_ly
                F_yval_before = fixed_area + active_sum_l * y_val - active_sum_ly
                
                if F_prev <= half_area <= F_yval_before:
                    if active_sum_l == 0:
                        return prev_y
                    else:
                        h_sol = (half_area + active_sum_ly - fixed_area) / active_sum_l
                        return h_sol
            
            events_here = events_dict[y_val]
            for event in events_here:
                typ, l_val, y_i = event
                if typ == 'start':
                    active_sum_l += l_val
                    active_sum_ly += l_val * y_i
                else:
                    active_sum_l -= l_val
                    active_sum_ly -= l_val * y_i
                    fixed_area += l_val * l_val
            
            prev_y = y_val
        
        return prev_y