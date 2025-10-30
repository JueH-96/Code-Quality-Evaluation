import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = int(data[1])
    B = int(data[2])
    D_list = list(map(int, data[3:3+n]))
    T = A + B
    
    bad_intervals = []
    
    for d in D_list:
        r = d % T
        if r < A:
            s1 = A - r
            e1 = T - r - 1
            bad_intervals.append((s1, e1))
        else:
            s1 = 0
            e1 = T - r - 1
            bad_intervals.append((s1, e1))
            s2 = T - r + A
            if s2 < T:
                bad_intervals.append((s2, T - 1))
    
    if not bad_intervals:
        print("Yes")
        return
        
    bad_intervals.sort(key=lambda x: x[0])
    
    merged = []
    start_curr, end_curr = bad_intervals[0]
    for i in range(1, len(bad_intervals)):
        s, e = bad_intervals[i]
        if s <= end_curr + 1:
            if e > end_curr:
                end_curr = e
        else:
            merged.append((start_curr, end_curr))
            start_curr, end_curr = s, e
    merged.append((start_curr, end_curr))
    
    if merged[0][0] > 0:
        print("Yes")
        return
        
    if merged[-1][1] < T - 1:
        print("Yes")
        return
        
    for i in range(1, len(merged)):
        if merged[i-1][1] + 1 < merged[i][0]:
            print("Yes")
            return
            
    print("No")

if __name__ == '__main__':
    main()