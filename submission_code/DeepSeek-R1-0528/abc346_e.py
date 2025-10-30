import sys
import bisect
from collections import defaultdict

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    first_line = data[0].split()
    H = int(first_line[0])
    W = int(first_line[1])
    M = int(first_line[2])
    
    row_time = [-1] * (H+1)
    row_color = [0] * (H+1)
    col_time = [-1] * (W+1)
    col_color = [0] * (W+1)
    
    for idx in range(M):
        t, a, x = map(int, data[1+idx].split())
        if t == 1:
            row_time[a] = idx
            row_color[a] = x
        else:
            col_time[a] = idx
            col_color[a] = x

    col_times_list = [col_time[j] for j in range(1, W+1)]
    col_times_list.sort()
    
    row_times_list = [row_time[i] for i in range(1, H+1)]
    row_times_list.sort()
    
    ans = defaultdict(int)
    
    for i in range(1, H+1):
        if row_time[i] == -1:
            continue
        t_i = row_time[i]
        pos = bisect.bisect_left(col_times_list, t_i)
        ans[row_color[i]] += pos
        
    for j in range(1, W+1):
        t_j = col_time[j]
        count_j = bisect.bisect_right(row_times_list, t_j)
        ans[col_color[j]] += count_j
        
    result = []
    for color, count in ans.items():
        if count > 0:
            result.append((color, count))
            
    result.sort(key=lambda x: x[0])
    
    print(len(result))
    for color, count in result:
        print(f"{color} {count}")

if __name__ == '__main__':
    main()