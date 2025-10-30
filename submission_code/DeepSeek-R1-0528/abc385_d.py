import sys
import bisect
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it)); m = int(next(it)); s_x = int(next(it)); s_y = int(next(it))
    houses = []
    for _ in range(n):
        x = int(next(it)); y = int(next(it))
        houses.append((x, y))
    
    moves = []
    for _ in range(m):
        d = next(it); c = int(next(it))
        moves.append((d, c))
    
    horizontal_groups = defaultdict(list)
    vertical_groups = defaultdict(list)
    cur_x, cur_y = s_x, s_y
    
    for d, c in moves:
        if d == 'U':
            next_x, next_y = cur_x, cur_y + c
        elif d == 'D':
            next_x, next_y = cur_x, cur_y - c
        elif d == 'R':
            next_x, next_y = cur_x + c, cur_y
        elif d == 'L':
            next_x, next_y = cur_x - c, cur_y
        
        if cur_x == next_x:
            y_low = min(cur_y, next_y)
            y_high = max(cur_y, next_y)
            vertical_groups[cur_x].append((y_low, y_high))
        else:
            x_low = min(cur_x, next_x)
            x_high = max(cur_x, next_x)
            horizontal_groups[cur_y].append((x_low, x_high))
        
        cur_x, cur_y = next_x, next_y
    
    final_x, final_y = cur_x, cur_y
    
    horizontal_dict = {}
    for y, intervals in horizontal_groups.items():
        if not intervals:
            continue
        intervals.sort()
        merged = []
        low, high = intervals[0]
        for i in range(1, len(intervals)):
            a, b = intervals[i]
            if a <= high:
                high = max(high, b)
            else:
                merged.append((low, high))
                low, high = a, b
        merged.append((low, high))
        starts = [inter[0] for inter in merged]
        horizontal_dict[y] = (merged, starts)
    
    vertical_dict = {}
    for x, intervals in vertical_groups.items():
        if not intervals:
            continue
        intervals.sort()
        merged = []
        low, high = intervals[0]
        for i in range(1, len(intervals)):
            a, b = intervals[i]
            if a <= high:
                high = max(high, b)
            else:
                merged.append((low, high))
                low, high = a, b
        merged.append((low, high))
        starts = [inter[0] for inter in merged]
        vertical_dict[x] = (merged, starts)
    
    count = 0
    for x, y in houses:
        found = False
        if y in horizontal_dict:
            merged_intervals, starts = horizontal_dict[y]
            pos = bisect.bisect_right(starts, x) - 1
            if pos >= 0:
                low_i, high_i = merged_intervals[pos]
                if x <= high_i:
                    count += 1
                    found = True
        if not found and x in vertical_dict:
            merged_intervals, starts = vertical_dict[x]
            pos = bisect.bisect_right(starts, y) - 1
            if pos >= 0:
                low_i, high_i = merged_intervals[pos]
                if y <= high_i:
                    count += 1
    
    print(f"{final_x} {final_y} {count}")

if __name__ == "__main__":
    main()