import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    M = int(input[idx]); idx += 1
    K = int(input[idx]); idx += 1
    A = list(map(int, input[idx:idx+N]))
    sum_A = sum(A)
    sorted_A = sorted(A)
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + sorted_A[i]
    
    output = []
    
    for i in range(N):
        Ai = A[i]
        low = 0
        high = K - sum_A + 1
        answer = -1
        
        while low <= high:
            mid = (low + high) // 2
            remaining = K - (sum_A + mid)
            if remaining < 0:
                high = mid - 1
                continue
            
            S = Ai + mid
            pos = bisect.bisect_right(sorted_A, S)
            current_excess_total = N - pos
            
            if Ai > S:
                current_excess = current_excess_total - 1
            else:
                current_excess = current_excess_total
            
            if current_excess >= M:
                low = mid + 1
                continue
            
            available = M - 1 - current_excess
            if available < 0:
                low = mid + 1
                continue
            
            p = pos
            if Ai <= S:
                count_candidates = p - 1
            else:
                count_candidates = p
            
            low_t = 0
            high_t = count_candidates
            best_t = 0
            
            while low_t <= high_t:
                mid_t = (low_t + high_t) // 2
                if mid_t == 0:
                    required_cost = 0
                else:
                    if count_candidates == 0:
                        sum_globals = 0
                    else:
                        use_t = min(mid_t, count_candidates)
                        start_pos = p - use_t
                        if start_pos < 0:
                            start_pos = 0
                        sum_globals = prefix_sum[p] - prefix_sum[start_pos]
                    
                    if Ai <= S:
                        left = bisect.bisect_left(sorted_A, Ai)
                        right = bisect.bisect_right(sorted_A, Ai)
                        overlap = False
                        if left < p:
                            if right - 1 >= start_pos:
                                overlap = True
                        if overlap:
                            sum_a = sum_globals - Ai
                            use_t -= 1
                            if use_t <= 0:
                                sum_a = 0
                        else:
                            sum_a = sum_globals
                    else:
                        sum_a = sum_globals
                        use_t = min(mid_t, count_candidates)
                    
                    if use_t == 0:
                        required_cost = 0
                    else:
                        required_cost = (S + 1) * use_t - sum_a
                
                if required_cost <= remaining:
                    best_t = use_t
                    low_t = mid_t + 1
                else:
                    high_t = mid_t - 1
            
            bought_max = current_excess + best_t
            if bought_max <= M - 1:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        if answer != -1:
            output.append(str(answer))
        else:
            X = 0
            S = Ai + X
            remaining = K - (sum_A + X)
            if remaining < 0:
                output.append('-1')
                continue
            
            pos = bisect.bisect_right(sorted_A, S)
            current_excess_total = N - pos
            
            if Ai > S:
                current_excess = current_excess_total - 1
            else:
                current_excess = current_excess_total
            
            if current_excess >= M:
                output.append('-1')
                continue
            
            available = M - 1 - current_excess
            if available < 0:
                output.append('-1')
                continue
            
            p = pos
            if Ai <= S:
                count_candidates = p - 1
            else:
                count_candidates = p
            
            low_t = 0
            high_t = count_candidates
            best_t = 0
            
            while low_t <= high_t:
                mid_t = (low_t + high_t) // 2
                if mid_t == 0:
                    required_cost = 0
                else:
                    use_t = min(mid_t, count_candidates)
                    start_pos = p - use_t
                    if start_pos < 0:
                        start_pos = 0
                    sum_globals = prefix_sum[p] - prefix_sum[start_pos]
                    
                    if Ai <= S:
                        left = bisect.bisect_left(sorted_A, Ai)
                        right = bisect.bisect_right(sorted_A, Ai)
                        overlap = False
                        if left < p:
                            if right - 1 >= start_pos:
                                overlap = True
                        if overlap:
                            sum_a = sum_globals - Ai
                            use_t -= 1
                            if use_t <= 0:
                                sum_a = 0
                        else:
                            sum_a = sum_globals
                    else:
                        sum_a = sum_globals
                        use_t = min(mid_t, count_candidates)
                    
                    if use_t == 0:
                        required_cost = 0
                    else:
                        required_cost = (S + 1) * use_t - sum_a
                
                if required_cost <= remaining:
                    best_t = use_t
                    low_t = mid_t + 1
                else:
                    high_t = mid_t - 1
            
            bought_max = current_excess + best_t
            if bought_max <= M - 1:
                output.append('0')
            else:
                output.append('-1')
    
    print(' '.join(output))

if __name__ == '__main__':
    main()