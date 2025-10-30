import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    N = int(data[0])
    M = int(data[1])
    pieces = []
    index = 2
    R_set = set()
    C_set = set()
    D1_set = set()
    D2_set = set()
    
    for i in range(M):
        a = int(data[index])
        b = int(data[index+1])
        index += 2
        pieces.append((a, b))
        R_set.add(a)
        C_set.add(b)
        D1_set.add(a+b)
        D2_set.add(a-b)
    
    total_squares = N * N
    
    s1 = 0
    s1 += len(R_set) * N
    s1 += len(C_set) * N
    
    for d in D1_set:
        low = max(1, d - N)
        high = min(N, d-1)
        if low <= high:
            s1 += (high - low + 1)
    
    for d in D2_set:
        low = max(1, d+1)
        high = min(N, N + d)
        if low <= high:
            s1 += (high - low + 1)
    
    s2 = 0
    s2 += len(R_set) * len(C_set)
    
    count13 = 0
    for i in R_set:
        for d1 in D1_set:
            j = d1 - i
            if 1 <= j <= N:
                count13 += 1
    s2 += count13
    
    count14 = 0
    for i in R_set:
        for d2 in D2_set:
            j = i - d2
            if 1 <= j <= N:
                count14 += 1
    s2 += count14
    
    count23 = 0
    for j in C_set:
        for d1 in D1_set:
            i_val = d1 - j
            if 1 <= i_val <= N:
                count23 += 1
    s2 += count23
    
    count24 = 0
    for j in C_set:
        for d2 in D2_set:
            i_val = j + d2
            if 1 <= i_val <= N:
                count24 += 1
    s2 += count24
    
    count34 = 0
    for d1 in D1_set:
        for d2 in D2_set:
            total_val = d1 + d2
            if total_val % 2 != 0:
                continue
            i_val = total_val // 2
            j_val = d1 - i_val
            if 1 <= i_val <= N and 1 <= j_val <= N:
                count34 += 1
    s2 += count34
    
    s3 = 0
    count123 = 0
    for i in R_set:
        for j in C_set:
            if (i + j) in D1_set:
                count123 += 1
    s3 += count123
    
    count124 = 0
    for i in R_set:
        for j in C_set:
            if (i - j) in D2_set:
                count124 += 1
    s3 += count124
    
    count134 = 0
    for i in R_set:
        for d1 in D1_set:
            j = d1 - i
            if 1 <= j <= N:
                d2 = i - j
                if d2 in D2_set:
                    count134 += 1
    s3 += count134
    
    count234 = 0
    for j in C_set:
        for d2 in D2_set:
            i_val = j + d2
            if 1 <= i_val <= N:
                d1 = i_val + j
                if d1 in D1_set:
                    count234 += 1
    s3 += count234
    
    s4 = 0
    for i in R_set:
        for j in C_set:
            if (i + j) in D1_set and (i - j) in D2_set:
                s4 += 1
    
    attacked = s1 - s2 + s3 - s4
    safe_squares = total_squares - attacked
    print(safe_squares)

if __name__ == "__main__":
    main()