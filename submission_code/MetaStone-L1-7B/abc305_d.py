import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    
    N = int(data[idx])
    idx += 1
    
    A = list(map(int, data[idx:idx + 2 * N]))
    idx += 2 * N
    
    M = (N - 1) // 2
    intervals = []
    for i in range(M):
        s = A[2 * i + 1]
        e = A[2 * i + 2]
        intervals.append((s, e))
    
    # Precompute prefix sums for e and s
    prefix_e_sum = [0] * (M + 1)
    prefix_s_sum = [0] * (M + 1)
    for i in range(M):
        prefix_e_sum[i + 1] = prefix_e_sum[i] + intervals[i][1]
        prefix_s_sum[i + 1] = prefix_s_sum[i] + intervals[i][0]
    
    Q = int(data[idx])
    idx += 1
    
    for _ in range(Q):
        l = int(data[idx])
        r = int(data[idx + 1])
        idx += 2
        
        # Compute a
        e_array = [e for s, e in intervals]
        left_e = bisect.bisect_left(e_array, l)
        low = left_e
        high = M - 1
        res_a = -1
        while low <= high:
            mid = (low + high) // 2
            if intervals[mid][0] <= r:
                res_a = mid
                high = mid - 1
            else:
                low = mid + 1
        a = res_a if res_a != -1 else None
        
        # Compute b
        right_e = bisect.bisect_left(e_array, l) - 1
        if right_e < 0:
            b = None
        else:
            low_b = 0
            high_b = right_e
            res_b = -1
            while low_b <= high_b:
                mid = (low_b + high_b) // 2
                if intervals[mid][0] <= r:
                    res_b = mid
                    low_b = mid + 1
                else:
                    high_b = mid - 1
            b = res_b if res_b != -1 else None
        
        if a is None or b is None or a > b:
            print(0)
            continue
        
        # Compute sum_e_less_equal_r
        j = bisect.bisect_right(e_array, r) - 1
        j = min(j, b)
        count_j = j - a + 1 if a <= j else 0
        sum_e_less_equal_r = prefix_e_sum[j + 1] - (prefix_e_sum[a] if a >= 0 else 0)
        count_e_greater_r = (b - a + 1) - count_j
        sum_e_r = sum_e_less_equal_r + count_e_greater_r * r
        
        # Compute sum_s_greater_or_equal_l
        s_array = [s for s, e in intervals]
        k = bisect.bisect_left(s_array, l, a, b + 1)
        count_k = (b - k + 1) if k <= b else 0
        sum_s_greater_or_equal_l = prefix_s_sum[b + 1] - (prefix_s_sum[k] if k >= 0 else 0)
        count_s_less_l = (b - a + 1) - count_k
        sum_s_l = sum_s_greater_or_equal_l + count_s_less_l * l
        
        total = sum_e_r - sum_s_l
        print(total)

if __name__ == '__main__':
    main()