def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    T0 = []
    T1 = []
    T2 = []
    for _ in range(N):
        t = int(input[idx])
        x = int(input[idx + 1])
        idx += 2
        if t == 0:
            T0.append(x)
        elif t == 1:
            T1.append(x)
        else:
            T2.append(x)
    
    # Sort and compute prefix sums for each type
    T0.sort(reverse=True)
    pre_T0 = [0]
    for x in T0:
        pre_T0.append(pre_T0[-1] + x)
    
    T1.sort(reverse=True)
    pre_T1 = [0]
    for x in T1:
        pre_T1.append(pre_T1[-1] + x)
    
    T2.sort(reverse=True)
    pre_T2 = [0]
    for x in T2:
        pre_T2.append(pre_T2[-1] + x)
    
    # Merge T0 and T1 into a single list and compute prefix sums
    merged = []
    i0 = i1 = 0
    while i0 < len(T0) or i1 < len(T1):
        if i0 < len(T0) and (i1 >= len(T1) or T0[i0] > T1[i1]):
            merged.append(T0[i0])
            i0 += 1
        else:
            merged.append(T1[i1])
            i1 += 1
    pre_merged = [0]
    for x in merged:
        pre_merged.append(pre_merged[-1] + x)
    
    max_total = 0
    len_T0 = len(T0)
    len_T1 = len(T1)
    len_T2 = len(T2)
    
    # Try all possible number of T2s to take (b)
    for b in range(0, len_T2 + 1):
        if b > M:
            continue
        remaining = M - b
        if remaining < 0:
            continue
        # Sum of T2's X for first b T2s
        S = pre_T2[b]
        C = remaining
        
        if S >= C:
            # All C items can be used (case1)
            if C >= len(pre_merged):
                sum_case = pre_merged[-1]
            else:
                sum_case = pre_merged[C]
            if sum_case > max_total:
                max_total = sum_case
        else:
            # Case2: sum of T1 used is min(S, c)
            part_a = 0
            part_a += pre_T0[min(C - S, len_T0)]
            part_a += pre_T1[min(S, len_T1)]
            
            # Compute part_b: max sum of a T0 and (C -a) T1, where (C -a) <= S-1
            part_b = 0
            a_start = max(0, C - (S - 1))
            a_end = min(C, len_T0)
            for a in range(a_start, a_end + 1):
                k = C - a
                if k < 0:
                    continue
                if k > S - 1:
                    k_val = pre_T1[min(S - 1, len_T1)]
                else:
                    k_val = pre_T1[k] if k <= len_T1 else pre_T1[len_T1]
                current = pre_T0[a] + k_val
                if current > part_b:
                    part_b = current
            
            current_max = max(part_a, part_b)
            if current_max > max_total:
                max_total = current_max
    
    print(max_total)

if __name__ == '__main__':
    main()