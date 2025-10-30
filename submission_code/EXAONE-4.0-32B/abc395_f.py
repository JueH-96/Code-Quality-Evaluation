def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    X = int(data[1])
    teeth = []
    total_sum = 0
    index = 2
    for i in range(n):
        u = int(data[index])
        d = int(data[index+1])
        index += 2
        teeth.append((u, d))
        total_sum += u + d
        
    M = min(u + d for u, d in teeth)
    
    def feasible(H):
        intervals = []
        for u, d in teeth:
            L_val = max(0, H - d)
            R_val = min(u, H)
            if L_val > R_val:
                return False
            intervals.append((L_val, R_val))
            
        a, b = intervals[0]
        for i in range(1, n):
            L_curr, R_curr = intervals[i]
            a_next = max(L_curr, a - X)
            b_next = min(R_curr, b + X)
            if a_next > b_next:
                return False
            a, b = a_next, b_next
        return True
        
    lo, hi = 0, M
    best_H = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            best_H = mid
            lo = mid + 1
        else:
            hi = mid - 1
            
    total_cost = total_sum - n * best_H
    print(total_cost)

if __name__ == "__main__":
    main()