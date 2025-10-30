import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    
    n = int(data[0])
    m = int(data[1])
    R_set = set()
    C_set = set()
    D1_set = set()
    D2_set = set()
    idx = 2
    for i in range(m):
        a = int(data[idx])
        b = int(data[idx+1])
        idx += 2
        R_set.add(a)
        C_set.add(b)
        D1_set.add(a + b)
        D2_set.add(a - b)
    
    lenR = len(R_set)
    lenC = len(C_set)
    lenD1 = len(D1_set)
    lenD2 = len(D2_set)
    
    setA = lenR * n
    setB = lenC * n
    
    setC = 0
    for s in D1_set:
        if s <= n + 1:
            setC += s - 1
        else:
            setC += 2 * n + 1 - s
            
    setD = 0
    for d in D2_set:
        ad = abs(d)
        setD += n - ad
        
    AB = lenR * lenC
    
    AC = 0
    for x in R_set:
        for s in D1_set:
            y = s - x
            if 1 <= y <= n:
                AC += 1
                
    AD = 0
    for x in R_set:
        for d in D2_set:
            y = x - d
            if 1 <= y <= n:
                AD += 1
                
    BC = 0
    for y in C_set:
        for s in D1_set:
            x = s - y
            if 1 <= x <= n:
                BC += 1
                
    BD = 0
    for y in C_set:
        for d in D2_set:
            x = y + d
            if 1 <= x <= n:
                BD += 1
                
    CD = 0
    for s in D1_set:
        for d in D2_set:
            if (s + d) % 2 == 0:
                x = (s + d) // 2
                y = (s - d) // 2
                if 1 <= x <= n and 1 <= y <= n:
                    CD += 1
                    
    ABC = 0
    for x in R_set:
        for y in C_set:
            if (x + y) in D1_set:
                ABC += 1
                
    ABD = 0
    for x in R_set:
        for y in C_set:
            if (x - y) in D2_set:
                ABD += 1
                
    ACD = 0
    for x in R_set:
        for s in D1_set:
            y = s - x
            if 1 <= y <= n:
                d_val = x - y
                if d_val in D2_set:
                    ACD += 1
                    
    BCD = 0
    for y in C_set:
        for s in D1_set:
            x = s - y
            if 1 <= x <= n:
                d_val = x - y
                if d_val in D2_set:
                    BCD += 1
                    
    ABCD = 0
    for x in R_set:
        for y in C_set:
            if (x + y) in D1_set and (x - y) in D2_set:
                ABCD += 1
                
    union_size = setA + setB + setC + setD - (AB + AC + AD + BC + BD + CD) + (ABC + ABD + ACD + BCD) - ABCD
    ans = n * n - union_size
    print(ans)

if __name__ == "__main__":
    main()