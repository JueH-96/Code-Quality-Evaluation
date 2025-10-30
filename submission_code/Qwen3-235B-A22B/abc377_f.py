import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    r_set = set()
    c_set = set()
    s_set = set()
    d_set = set()
    
    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        r_set.add(a)
        c_set.add(b)
        s_set.add(a + b)
        d_set.add(a - b)
    
    R = list(r_set)
    C = list(c_set)
    S = list(s_set)
    D = list(d_set)
    
    R_size = len(R)
    C_size = len(C)
    S_size = len(S)
    D_size = len(D)
    
    A = R_size * N
    B = C_size * N
    
    sum_C = 0
    for s in S:
        if s <= N + 1:
            cnt = s - 1
        else:
            cnt = 2 * N + 1 - s
        if cnt < 0:
            cnt = 0
        sum_C += cnt
    
    sum_D = 0
    for d in D:
        ad = abs(d)
        if ad >= N:
            cnt = 0
        else:
            cnt = N - ad
        if cnt < 0:
            cnt = 0
        sum_D += cnt
    
    AB = R_size * C_size
    
    AC = 0
    for r in R:
        for s in S:
            y = s - r
            if 1 <= y <= N:
                AC += 1
    
    AD = 0
    for r in R:
        for d in D:
            x = r + d
            if 1 <= x <= N:
                AD += 1
    
    BC = 0
    for c in C:
        for s in S:
            x = s - c
            if 1 <= x <= N:
                BC += 1
    
    BD = 0
    for c in C:
        for d in D:
            x = c + d
            if 1 <= x <= N:
                BD += 1
    
    CD = 0
    for s in S:
        for d in D:
            if (s + d) % 2 != 0:
                continue
            x = (s + d) // 2
            y = (s - d) // 2
            if 1 <= x <= N and 1 <= y <= N:
                CD += 1
    
    ABC = 0
    for r in R:
        for c in C:
            if (r + c) in s_set:
                ABC += 1
    
    ABD = 0
    for r in R:
        for c in C:
            if (r - c) in d_set:
                ABD += 1
    
    ACD = 0
    for r in R:
        for s in S:
            y = s - r
            if 1 <= y <= N:
                d_candidate = 2 * r - s
                if d_candidate in d_set:
                    ACD += 1
    
    BCD = 0
    for c in C:
        for d in D:
            x = c + d
            if 1 <= x <= N:
                s_candidate = 2 * c + d
                if s_candidate in s_set:
                    BCD += 1
    
    ABCD = 0
    for r in R:
        for c in C:
            if (r + c) in s_set and (r - c) in d_set:
                ABCD += 1
    
    forbidden = A + B + sum_C + sum_D
    forbidden -= (AB + AC + AD + BC + BD + CD)
    forbidden += (ABC + ABD + ACD + BCD)
    forbidden -= ABCD
    
    ans = N * N - forbidden
    print(ans)

if __name__ == "__main__":
    main()