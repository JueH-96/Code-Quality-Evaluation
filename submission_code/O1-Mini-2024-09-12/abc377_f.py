import sys
import bisect

def solve():
    import sys
    import sys
    from bisect import bisect_left, bisect_right

    def readints():
        return list(map(int, sys.stdin.read().split()))
    
    data = readints()
    N = data[0]
    M = data[1]
    a_b = data[2:]
    R = set()
    C = set()
    D1 = set()
    D2 = set()
    for k in range(M):
        a = a_b[2*k]
        b = a_b[2*k+1]
        R.add(a)
        C.add(b)
        D1.add(a - b)
        D2.add(a + b)
    
    sorted_R = sorted(R)
    sorted_C = sorted(C)
    sorted_D1 = sorted(D1)
    sorted_D2 = sorted(D2)
    
    def count_in_range(sorted_list, low, high):
        if not sorted_list:
            return 0
        left = bisect.bisect_left(sorted_list, low)
        right = bisect.bisect_right(sorted_list, high)
        return right - left

    # Precompute list representations for faster access in overlaps
    list_R = sorted_R
    list_C = sorted_C

    # Function to get elements in R within [low, high]
    def get_R_in_range(low, high):
        left = bisect_left(list_R, low)
        right = bisect_right(list_R, high)
        return list_R[left:right]

    # Function to get elements in C within [low, high]
    def get_C_in_range(low, high):
        left = bisect_left(list_C, low)
        right = bisect_right(list_C, high)
        return list_C[left:right]

    # Compute sum of c_d1
    total_c_d1 = 0
    for d1 in D1:
        max_i = max(1, 1 + d1)
        min_i = min(N, N + d1)
        if min_i < max_i:
            continue
        total = min_i - max_i + 1

        count_R = count_in_range(sorted_R, max_i, min_i)

        j_min = max(1, max_i - d1)
        j_max = min(N, min_i - d1)
        if j_max < j_min:
            count_C = 0
        else:
            count_C = count_in_range(sorted_C, j_min, j_max)
        
        # Compute overlap: number of i in R within [max_i, min_i] and j = i - d1 in C
        overlap_count = 0
        R_in_range = get_R_in_range(max_i, min_i)
        for i in R_in_range:
            j = i - d1
            if 1 <= j <= N and j in C:
                overlap_count +=1
        
        c_d1 = total - count_R - count_C + overlap_count
        total_c_d1 += c_d1

    # Compute sum of c_d2
    total_c_d2 = 0
    for d2 in D2:
        max_i = max(1, d2 - N)
        min_i = min(N, d2 -1)
        if min_i < max_i:
            continue
        total = min_i - max_i +1

        count_R = count_in_range(sorted_R, max_i, min_i)

        j_min = max(1, d2 - min_i)
        j_max = min(N, d2 - max_i)
        if j_max < j_min:
            count_C = 0
        else:
            count_C = count_in_range(sorted_C, j_min, j_max)
        
        # Compute overlap: number of i in R within [max_i, min_i] and j = d2 -i in C
        overlap_count = 0
        R_in_range = get_R_in_range(max_i, min_i)
        for i in R_in_range:
            j = d2 - i
            if 1 <= j <= N and j in C:
                overlap_count +=1
        
        c_d2 = total - count_R - count_C + overlap_count
        total_c_d2 += c_d2

    # Compute sum of c_d1_d2
    total_c_d1_d2 = 0
    for d1 in D1:
        for d2 in D2:
            if (d1 + d2) %2 !=0:
                continue
            i = (d1 + d2) //2
            j = (d2 - d1) //2
            if 1 <= i <= N and 1 <= j <= N:
                if i not in R and j not in C:
                    total_c_d1_d2 +=1

    # Compute safe squares
    safe = (N - len(R)) * (N - len(C))
    safe -= total_c_d1
    safe -= total_c_d2
    safe += total_c_d1_d2

    print(safe)