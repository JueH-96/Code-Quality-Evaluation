import sys
import bisect

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    n, q = map(int, data[0].split())
    s = data[1].strip()
    queries = []
    for i in range(q):
        parts = data[2+i].split()
        if len(parts) < 2:
            continue
        L = int(parts[0])
        R = int(parts[1])
        queries.append((L, R))
    
    F1 = [0] * (n+1)
    F2 = [0] * (n+1)
    for i in range(1, n+1):
        char = s[i-1]
        F1[i] = F1[i-1] + (1 if char == '1' else 0)
        F2[i] = F2[i-1] + (1 if char == '2' else 0)
    
    centers_list = []
    for i in range(n):
        if s[i] == '/':
            centers_list.append(i)
    
    out_lines = []
    for (L, R) in queries:
        l0 = L - 1
        r0 = R - 1
        left_pos = bisect.bisect_left(centers_list, l0)
        right_pos = bisect.bisect_right(centers_list, r0) - 1
        
        if left_pos > right_pos:
            out_lines.append("0")
            continue
            
        X = F1[l0]
        Y = F2[r0+1]
        
        lo = left_pos
        hi = right_pos
        best_f = -10**9
        
        while hi - lo >= 3:
            m1 = lo + (hi - lo) // 3
            m2 = hi - (hi - lo) // 3
            i1 = centers_list[m1]
            i2 = centers_list[m2]
            
            ones_left1 = F1[i1] - X
            if ones_left1 < 0:
                ones_left1 = 0
            twos_right1 = Y - F2[i1+1]
            if twos_right1 < 0:
                twos_right1 = 0
            f1 = min(ones_left1, twos_right1)
            
            ones_left2 = F1[i2] - X
            if ones_left2 < 0:
                ones_left2 = 0
            twos_right2 = Y - F2[i2+1]
            if twos_right2 < 0:
                twos_right2 = 0
            f2 = min(ones_left2, twos_right2)
            
            if f1 < f2:
                lo = m1 + 1
            else:
                hi = m2 - 1
                
        for idx in range(lo, hi+1):
            i_center = centers_list[idx]
            ones_left = F1[i_center] - X
            if ones_left < 0:
                ones_left = 0
            twos_right = Y - F2[i_center+1]
            if twos_right < 0:
                twos_right = 0
            f_val = min(ones_left, twos_right)
            if f_val > best_f:
                best_f = f_val
                
        ans = 2 * best_f + 1
        out_lines.append(str(ans))
        
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()