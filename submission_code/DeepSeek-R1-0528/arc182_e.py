import math
import bisect

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    C = int(data[2])
    K = int(data[3])
    A = list(map(int, input().split()))
    
    if C == 0:
        min_val = min(A)
        print(min_val * K)
        return
        
    g = math.gcd(C, M)
    T_val = M // g
    c_val = C // g
    
    D = sorted(set(A))
    n = len(D)
    boundaries = [d // g for d in D]
    
    S_cycle = 0
    
    a0 = 0
    b0 = boundaries[0]
    count0 = b0 - a0 + 1
    sum_j0 = (a0 + b0) * count0 // 2
    gap0 = D[0] * count0 - g * sum_j0
    S_cycle += gap0
    
    for i in range(1, n):
        a_seg = boundaries[i-1] + 1
        b_seg = boundaries[i]
        if a_seg > b_seg:
            count_seg = 0
            sum_j_seg = 0
        else:
            count_seg = b_seg - a_seg + 1
            sum_j_seg = (a_seg + b_seg) * count_seg // 2
        gap_seg = D[i] * count_seg - g * sum_j_seg
        S_cycle += gap_seg
        
    a_last = boundaries[-1] + 1
    b_last = T_val - 1
    if a_last <= b_last:
        count_last = b_last - a_last + 1
        sum_j_last = (a_last + b_last) * count_last // 2
        gap_last = (D[0] + M) * count_last - g * sum_j_last
        S_cycle += gap_last
        
    full_cycles = K // T_val
    rem = K % T_val
    
    total_rem = 0
    if rem > 0:
        if rem <= 10**6 or T_val <= 10**6:
            for k in range(rem):
                r = (c_val * k) % T_val
                if r == 0:
                    j_k = 0
                else:
                    j_k = T_val - r
                pos = bisect.bisect_left(boundaries, j_k)
                if pos < n:
                    gap = D[pos] - g * j_k
                else:
                    gap = D[0] + M - g * j_k
                total_rem += gap
        else:
            total_rem = 0
            
    ans = full_cycles * S_cycle + total_rem
    print(ans)

if __name__ == '__main__':
    main()