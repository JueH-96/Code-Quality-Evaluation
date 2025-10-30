import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    
    blacks = []
    whites = []
    index = 2
    for _ in range(m):
        x = int(data[index])
        y = int(data[index+1])
        c = data[index+2]
        index += 3
        if c == 'B':
            blacks.append((x, y))
        elif c == 'W':
            whites.append((x, y))
            
    critical_rows_set = {1, n}
    for x, y, c in zip(*[iter(data[2:])]*3):
        x = int(x)
        critical_rows_set.add(x)
    critical_rows_asc = sorted(critical_rows_set)
    critical_rows_desc = sorted(critical_rows_set, reverse=True)
    
    L_vals = {}
    if blacks:
        blacks_sorted = sorted(blacks, key=lambda p: (-p[0], -p[1]))
    else:
        blacks_sorted = []
    current_max = 0
    ptr = 0
    for r in critical_rows_desc:
        while ptr < len(blacks_sorted) and blacks_sorted[ptr][0] >= r:
            current_max = max(current_max, blacks_sorted[ptr][1])
            ptr += 1
        L_vals[r] = current_max
        
    U_vals = {}
    if whites:
        whites_sorted = sorted(whites, key=lambda p: (p[0], p[1]))
    else:
        whites_sorted = []
    current_min = n + 1
    ptrw = 0
    for r in critical_rows_asc:
        while ptrw < len(whites_sorted) and whites_sorted[ptrw][0] <= r:
            current_min = min(current_min, whites_sorted[ptrw][1])
            ptrw += 1
        U_vals[r] = current_min
    if not whites:
        for r in critical_rows_asc:
            U_vals[r] = n + 1
            
    low_bound = 0
    high_bound = n
    for r in critical_rows_desc:
        L_r = L_vals.get(r, 0)
        U_r = U_vals.get(r, n + 1)
        high_candidate = min(high_bound, U_r - 1)
        low_candidate = max(low_bound, L_r)
        if low_candidate > high_candidate:
            print("No")
            return
        a_r = low_candidate
        low_bound = a_r
        
    print("Yes")

if __name__ == "__main__":
    main()