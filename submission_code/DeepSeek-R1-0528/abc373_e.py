import bisect

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:3+n]))
    
    total_A = sum(A)
    R_total = K - total_A
    
    B = sorted(A)
    n_val = n
    S = [0] * n_val
    if n_val > 0:
        S[0] = B[0]
        for i in range(1, n_val):
            S[i] = S[i-1] + B[i]
    
    C = [(A[i], i) for i in range(n_val)]
    C_sorted = sorted(C, key=lambda x: (x[0], x[1]))
    pos_arr = [0] * n_val
    for idx, (val, orig_idx) in enumerate(C_sorted):
        pos_arr[orig_idx] = idx
    
    candidate_ans = [0] * n_val
    
    for i in range(n_val):
        v_max = A[i] + R_total
        total_count_max = bisect.bisect_right(B, v_max)
        if A[i] <= v_max:
            count_others_below = total_count_max - 1
        else:
            count_others_below = total_count_max
        count_above = (n_val - 1) - count_others_below
        if count_above >= M:
            candidate_ans[i] = -1
            continue
        
        low = 0
        high = R_total
        ans_X = None
        while low <= high:
            mid = (low + high) // 2
            v = A[i] + mid
            total_count = bisect.bisect_right(B, v)
            
            if A[i] <= v:
                total_count_i = total_count - 1
            else:
                total_count_i = total_count
                
            c = (n_val - 1) - total_count_i
            k0 = M - (n_val - 1) + total_count_i
            
            if k0 <= 0:
                T_val = 0
            elif k0 > total_count_i:
                T_val = R_total + 1
            else:
                start = total_count - k0
                end = total_count - 1
                if start > end:
                    total_topk_sum = 0
                else:
                    if start == 0:
                        total_topk_sum = S[end]
                    else:
                        total_topk_sum = S[end] - S[start-1]
                
                if A[i] <= v and start <= pos_arr[i] <= end:
                    start2 = total_count - k0 - 1
                    if start2 < 0:
                        start2 = 0
                    if start2 > end:
                        total_topk_plus1_sum = total_topk_sum
                    else:
                        if start2 == 0:
                            total_topk_plus1_sum = S[end]
                        else:
                            total_topk_plus1_sum = S[end] - S[start2-1]
                    sum_i = total_topk_plus1_sum - A[i]
                else:
                    sum_i = total_topk_sum
                
                T_val = k0 * (v + 1) - sum_i
                
            if R_total - mid < T_val:
                ans_X = mid
                high = mid - 1
            else:
                low = mid + 1
        
        candidate_ans[i] = ans_X
    
    print(" ".join(str(x) for x in candidate_ans))

if __name__ == '__main__':
    main()