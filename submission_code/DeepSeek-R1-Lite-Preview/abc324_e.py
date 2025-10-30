def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = data[1]
    K = len(T)
    S_list = data[2:2+N]
    
    # Compute match_i for each S_i
    match_i = []
    for s in S_list:
        t_ptr = 0
        for c in s:
            if c == T[t_ptr]:
                t_ptr += 1
                if t_ptr == K:
                    break
        match_i.append(t_ptr)
    
    # Compute match_j for each S_j
    match_j = []
    for s in S_list:
        t_ptr = K - 1
        for c in reversed(s):
            if c == T[t_ptr]:
                t_ptr -= 1
                if t_ptr == -1:
                    break
        if t_ptr < 0:
            match_j.append(K)
        else:
            match_j.append(K - 1 - t_ptr)
    
    # Compute A[m] = number of Si with match_i >= m
    A = [0] * (K + 1)
    for m in range(K + 1):
        count = 0
        for mi in match_i:
            if mi >= m:
                count += 1
        A[m] = count
    
    # Compute B[m] = number of Sj with match_j >= m
    B = [0] * (K + 1)
    for m in range(K + 1):
        count = 0
        for mj in match_j:
            if mj >= m:
                count += 1
        B[m] = count
    
    # Compute total number of valid pairs
    total = 0
    for m in range(K + 1):
        if K - m >= 0:
            total += A[m] * B[K - m]
    
    print(total)

if __name__ == "__main__":
    main()