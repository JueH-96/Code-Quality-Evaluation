from itertools import permutations
import math

def main():
    N, S, T = map(int, input().split())
    segments = []
    for _ in range(N):
        A, B, C, D = map(int, input().split())
        seg_start = (A, B)
        seg_end = (C, D)
        segments.append((seg_start, seg_end))
    
    lengths = []
    for seg in segments:
        seg_start, seg_end = seg
        dx = seg_end[0] - seg_start[0]
        dy = seg_end[1] - seg_start[1]
        lengths.append(math.hypot(dx, dy))
    
    min_time = float('inf')
    
    for perm in permutations(segments):
        sum_print = sum(l / T for l in lengths)
        for mask in range(0, 1 << N):
            sum_trans = 0.0
            # Determine the starting point of the first segment
            first_seg = perm[0]
            seg_start, seg_end = first_seg
            if (mask >> 0) & 1:
                current_start = seg_end
            else:
                current_start = seg_start
            prev_end = seg_end if current_start == seg_start else seg_start
            
            for i in range(1, N):
                current_seg = perm[i]
                seg_start_current, seg_end_current = current_seg
                
                if (mask >> i) & 1:
                    current_start_seg = seg_end_current
                else:
                    current_start_seg = seg_start_current
                
                dx = current_start_seg[0] - prev_end[0]
                dy = current_start_seg[1] - prev_end[1]
                distance = math.hypot(dx, dy)
                sum_trans += distance / S
                
                # Update prev_end for the next iteration
                if current_start_seg == seg_start_current:
                    prev_end = seg_start_current
                else:
                    prev_end = seg_end_current
            
            total_time = sum_print + sum_trans
            if total_time < min_time:
                min_time = total_time
    
    print("{0:.15f}".format(min_time))

if __name__ == '__main__':
    main()