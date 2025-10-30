import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    m = int(data[1])
    k = int(data[2])
    a = list(map(int, data[3:3+n]))
    sum_a = sum(a)
    r_remaining = k - sum_a
    
    if r_remaining < 0:
        print(' '.join(['-1'] * n))
        return
    
    sorted_candidates = sorted([(a[i], i) for i in range(n)])
    s = [x[0] for x in sorted_candidates]
    prefix_sum = [0] * (len(s) + 1)
    for i in range(len(s)):
        prefix_sum[i + 1] = prefix_sum[i] + s[i]
    
    pos_dict = {x[1]: idx for idx, x in enumerate(sorted_candidates)}
    
    result = []
    for i in range(n):
        ai = a[i]
        pos_i = pos_dict[i]
        
        low_x = 0
        high_x = r_remaining
        ans_x = -1
        
        while low_x <= high_x:
            mid_x = (low_x + high_x) // 2
            s_val = ai + mid_x
            r_prime = r_remaining - mid_x
            
            upper_idx = bisect.bisect_right(s, s_val)
            original_gt = len(s) - upper_idx
            if s[pos_i] > s_val:
                group1 = original_gt - 1
            else:
                group1 = original_gt
            
            original_le = bisect.bisect_right(s, s_val)
            if s[pos_i] <= s_val:
                merged_size = original_le - 1
            else:
                merged_size = original_le
            
            low_t = 0
            high_t = merged_size
            t_candidate = 0
            
            while low_t <= high_t:
                mid_t = (low_t + high_t) // 2
                start_merged = merged_size - mid_t
                end_merged = merged_size - 1
                sum_bj = 0
                
                if mid_t == 0:
                    sum_bj = 0
                else:
                    if merged_size == original_le - 1:
                        left_part_end = pos_i - 1
                        right_part_start = pos_i
                        right_part_end = merged_size - 1
                        
                        sum_left = 0
                        left_start = start_merged
                        left_end = min(end_merged, left_part_end)
                        if left_start <= left_end and left_start >= 0:
                            sum_left = prefix_sum[left_end + 1] - prefix_sum[left_start]
                        
                        right_start = max(start_merged, right_part_start)
                        right_end = end_merged
                        sum_right = 0
                        if right_start <= right_end and right_start <= right_part_end:
                            if right_end + 2 <= len(prefix_sum) - 1:
                                sum_right = prefix_sum[right_end + 2] - prefix_sum[right_start + 1]
                        
                        sum_bj = sum_left + sum_right
                    else:
                        if start_merged < 0:
                            sum_bj = 0
                        else:
                            if start_merged > end_merged:
                                sum_bj = 0
                            else:
                                sum_bj = prefix_sum[end_merged + 1] - prefix_sum[start_merged]
                
                sum_needed = mid_t * (s_val + 1) - sum_bj
                if sum_needed <= r_prime:
                    t_candidate = mid_t
                    low_t = mid_t + 1
                else:
                    high_t = mid_t - 1
            
            t_val = group1 + t_candidate
            if t_val < m:
                ans_x = mid_x
                high_x = mid_x - 1
            else:
                low_x = mid_x + 1
        
        if ans_x == -1:
            X_test = r_remaining
        else:
            X_test = ans_x
        
        s_val_test = ai + X_test
        r_prime_test = r_remaining - X_test
        
        upper_idx_test = bisect.bisect_right(s, s_val_test)
        original_gt_test = len(s) - upper_idx_test
        if s[pos_i] > s_val_test:
            group1_test = original_gt_test - 1
        else:
            group1_test = original_gt_test
        
        original_le_test = bisect.bisect_right(s, s_val_test)
        if s[pos_i] <= s_val_test:
            merged_size_test = original_le_test - 1
        else:
            merged_size_test = original_le_test
        
        low_t_test = 0
        high_t_test = merged_size_test
        t_candidate_test = 0
        
        while low_t_test <= high_t_test:
            mid_t = (low_t_test + high_t_test) // 2
            start_merged = merged_size_test - mid_t
            end_merged = merged_size_test - 1
            sum_bj = 0
            
            if mid_t == 0:
                sum_bj = 0
            else:
                if merged_size_test == original_le_test - 1:
                    left_part_end = pos_i - 1
                    right_part_start = pos_i
                    right_part_end = merged_size_test - 1
                    
                    sum_left = 0
                    left_start = start_merged
                    left_end = min(end_merged, left_part_end)
                    if left_start <= left_end and left_start >= 0:
                        sum_left = prefix_sum[left_end + 1] - prefix_sum[left_start]
                    
                    right_start = max(start_merged, right_part_start)
                    right_end = end_merged
                    sum_right = 0
                    if right_start <= right_end and right_start <= right_part_end:
                        sum_right = prefix_sum[right_end + 2] - prefix_sum[right_start + 1]
                    
                    sum_bj = sum_left + sum_right
                else:
                    if start_merged < 0:
                        sum_bj = 0
                    else:
                        if start_merged > end_merged:
                            sum_bj = 0
                        else:
                            sum_bj = prefix_sum[end_merged + 1] - prefix_sum[start_merged]
            
            sum_needed_test = mid_t * (s_val_test + 1) - sum_bj
            if sum_needed_test <= r_prime_test:
                t_candidate_test = mid_t
                low_t_test = mid_t + 1
            else:
                high_t_test = mid_t - 1
        
        t_val_test = group1_test + t_candidate_test
        
        if t_val_test < m:
            result.append(str(X_test))
        else:
            result.append('-1')
    
    print(' '.join(result))

if __name__ == '__main__':
    main()