import bisect

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    pos = [[] for _ in range(n+2)]
    
    for i in range(n):
        a = A[i]
        if 1 <= a <= n:
            pos[a].append(i)
    
    total_ans = 0
    
    for x in range(1, n+1):
        if not pos[x]:
            continue
            
        if x == 1:
            segments = [(0, n-1)]
        else:
            if not pos[x-1]:
                segments = [(0, n-1)]
            else:
                segments = []
                first_occ = pos[x-1][0]
                if first_occ > 0:
                    segments.append((0, first_occ-1))
                for idx in range(len(pos[x-1])-1):
                    start_seg = pos[x-1][idx] + 1
                    end_seg = pos[x-1][idx+1] - 1
                    if start_seg <= end_seg:
                        segments.append((start_seg, end_seg))
                last_occ = pos[x-1][-1]
                if last_occ < n-1:
                    segments.append((last_occ+1, n-1))
        
        for (a, b) in segments:
            if a > b:
                continue
            l_idx = bisect.bisect_left(pos[x], a)
            r_idx = bisect.bisect_right(pos[x], b) - 1
            if l_idx > r_idx:
                continue
                
            L_seg = b - a + 1
            total_sub = L_seg * (L_seg + 1) // 2
            first_in_seg = pos[x][l_idx]
            last_in_seg = pos[x][r_idx]
            gap0 = first_in_seg - a
            gap_last = b - last_in_seg
            gap_sum = gap0 * (gap0 + 1) // 2 + gap_last * (gap_last + 1) // 2
            
            for i in range(l_idx, r_idx):
                gap = pos[x][i+1] - pos[x][i] - 1
                gap_sum += gap * (gap + 1) // 2
                
            total_ans += total_sub - gap_sum
            
    print(total_ans)

if __name__ == "__main__":
    main()