import sys
import bisect

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    M = int(next(it))
    
    row_time = [0] * (H + 1)
    row_color = [0] * (H + 1)
    col_time = [0] * (W + 1)
    col_color = [0] * (W + 1)
    
    operations = []
    for _ in range(M):
        t = int(next(it))
        a = int(next(it))
        x = int(next(it))
        operations.append((t, a, x))
    
    for idx in range(M):
        t, a, x = operations[idx]
        time_index = idx + 1
        if t == 1:
            row_time[a] = time_index
            row_color[a] = x
        else:
            col_time[a] = time_index
            col_color[a] = x
    
    row_times_list = row_time[1:]
    col_times_list = col_time[1:]
    
    row_times_list.sort()
    col_times_list.sort()
    
    ans = {}
    
    for i in range(1, H + 1):
        t = row_time[i]
        if t == 0:
            continue
        pos = bisect.bisect_right(col_times_list, t)
        c = row_color[i]
        ans[c] = ans.get(c, 0) + pos
    
    for j in range(1, W + 1):
        t = col_time[j]
        if t == 0:
            continue
        pos = bisect.bisect_right(row_times_list, t)
        c = col_color[j]
        ans[c] = ans.get(c, 0) + pos
    
    R0 = row_times_list.count(0)
    C0 = col_times_list.count(0)
    total_unset = R0 * C0
    ans[0] = ans.get(0, 0) + total_unset
    
    distinct_colors = []
    for color, count in ans.items():
        if count > 0:
            distinct_colors.append(color)
    
    distinct_colors.sort()
    k = len(distinct_colors)
    print(k)
    for color in distinct_colors:
        print(f"{color} {ans[color]}")

if __name__ == "__main__":
    main()