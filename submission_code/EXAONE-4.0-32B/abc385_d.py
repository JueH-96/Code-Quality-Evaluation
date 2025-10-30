import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    sx = int(next(it))
    sy = int(next(it))
    
    houses = []
    for _ in range(n):
        x = int(next(it))
        y = int(next(it))
        houses.append((x, y))
        
    moves = []
    for _ in range(m):
        d = next(it)
        c = int(next(it))
        moves.append((d, c))
        
    current_x, current_y = sx, sy
    segments = []
    
    for d, c in moves:
        nx, ny = current_x, current_y
        if d == 'U':
            ny += c
        elif d == 'D':
            ny -= c
        elif d == 'L':
            nx -= c
        elif d == 'R':
            nx += c
        segments.append((current_x, current_y, nx, ny))
        current_x, current_y = nx, ny
        
    final_x, final_y = current_x, current_y
    
    horizontal_dict = {}
    vertical_dict = {}
    
    for seg in segments:
        x1, y1, x2, y2 = seg
        if y1 == y2:
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            if y1 not in horizontal_dict:
                horizontal_dict[y1] = []
            horizontal_dict[y1].append((x_min, x_max))
        elif x1 == x2:
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            if x1 not in vertical_dict:
                vertical_dict[x1] = []
            vertical_dict[x1].append((y_min, y_max))
    
    merged_horizontal = {}
    for y, intervals in horizontal_dict.items():
        intervals.sort(key=lambda inter: inter[0])
        merged = []
        start_curr, end_curr = intervals[0]
        for i in range(1, len(intervals)):
            s, e = intervals[i]
            if s <= end_curr + 1:
                end_curr = max(end_curr, e)
            else:
                merged.append((start_curr, end_curr))
                start_curr, end_curr = s, e
        merged.append((start_curr, end_curr))
        merged_horizontal[y] = merged
        
    merged_vertical = {}
    for x, intervals in vertical_dict.items():
        intervals.sort(key=lambda inter: inter[0])
        merged = []
        start_curr, end_curr = intervals[0]
        for i in range(1, len(intervals)):
            s, e = intervals[i]
            if s <= end_curr + 1:
                end_curr = max(end_curr, e)
            else:
                merged.append((start_curr, end_curr))
                start_curr, end_curr = s, e
        merged.append((start_curr, end_curr))
        merged_vertical[x] = merged
        
    count = 0
    for (x, y) in houses:
        found = False
        if y in merged_horizontal:
            intervals_list = merged_horizontal[y]
            lo, hi = 0, len(intervals_list) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                s, e = intervals_list[mid]
                if x < s:
                    hi = mid - 1
                elif x > e:
                    lo = mid + 1
                else:
                    found = True
                    break
        if found:
            count += 1
            continue
            
        if x in merged_vertical:
            intervals_list = merged_vertical[x]
            lo, hi = 0, len(intervals_list) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                s, e = intervals_list[mid]
                if y < s:
                    hi = mid - 1
                elif y > e:
                    lo = mid + 1
                else:
                    found = True
                    break
            if found:
                count += 1
                
    print(f"{final_x} {final_y} {count}")

if __name__ == "__main__":
    main()