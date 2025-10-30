def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    pieces = []
    for _ in range(M):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        pieces.append((a, b))
    
    rows = set()
    cols = set()
    sums = set()
    diffs = set()
    
    for a, b in pieces:
        rows.add(a)
        cols.add(b)
        sums.add(a + b)
        diffs.add(a - b)
    
    A = len(rows) * N
    B = len(cols) * N
    
    C = 0
    for s in sums:
        if s <= N + 1:
            C += s - 1
        else:
            C += 2 * N + 1 - s
    
    D = sum(N - abs(d) for d in diffs)
    
    AB = len(rows) * len(cols)
    
    AC = 0
    for a in rows:
        for s in sums:
            b = s - a
            if 1 <= b <= N:
                AC += 1
    
    AD = 0
    for a in rows:
        for d in diffs:
            b = a - d
            if 1 <= b <= N:
                AD += 1
    
    BC = 0
    for b in cols:
        for s in sums:
            a = s - b
            if 1 <= a <= N:
                BC += 1
    
    BD = 0
    for b in cols:
        for d in diffs:
            a = b + d
            if 1 <= a <= N:
                BD += 1
    
    CD = 0
    for s in sums:
        for d in diffs:
            if (s - d) % 2 != 0:
                continue
            x = (s + d) // 2
            y = (s - d) // 2
            if 1 <= x <= N and 1 <= y <= N:
                CD += 1
    
    ABC = 0
    sum_set = sums
    for a in rows:
        for b in cols:
            if (a + b) in sum_set:
                ABC += 1
    
    ABD = 0
    diff_set = diffs
    for a in rows:
        for b in cols:
            if (a - b) in diff_set:
                ABD += 1
    
    ACD = 0
    for a in rows:
        for d in diffs:
            s = 2 * a - d
            if s in sums:
                b = a - d
                if 1 <= b <= N:
                    ACD += 1
    
    BCD = 0
    for b in cols:
        for d in diffs:
            s = 2 * b + d
            if s in sums:
                a = b + d
                if 1 <= a <= N:
                    BCD += 1
    
    ABCD = 0
    for a in rows:
        for b in cols:
            if (a + b) in sums and (a - b) in diffs:
                ABCD += 1
    
    forbidden = (A + B + C + D 
                - (AB + AC + AD + BC + BD + CD) 
                + (ABC + ABD + ACD + BCD) 
                - ABCD)
    
    answer = N * N - forbidden
    print(answer)

if __name__ == "__main__":
    main()