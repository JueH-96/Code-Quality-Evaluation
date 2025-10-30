import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    K = int(input[ptr]); ptr +=1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    sumA = sum(A)
    remaining_total = K - sumA
    sorted_A = sorted(A)
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + sorted_A[i]
    
    res = []
    for i in range(N):
        original_A_i = A[i]
        lo_i = bisect.bisect_left(sorted_A, original_A_i)
        hi_i = bisect.bisect_right(sorted_A, original_A_i)
        
        # Calculate initial count
        pos_initial = bisect.bisect_right(sorted_A, original_A_i)
        initial_count = N - pos_initial
        if initial_count < M:
            res.append(0)
            continue
        
        # Check if possible with max_S
        max_S = original_A_i + remaining_total
        pos_S = bisect.bisect_right(sorted_A, max_S)
        global_count = N - pos_S
        if original_A_i > max_S:
            z = global_count - 1
        else:
            z = global_count
        t_global = pos_S
        if original_A_i <= max_S:
            t = t_global - 1
        else:
            t = t_global
        
        possible = False
        if z >= M:
            pass  # Not possible
        else:
            take = M - z
            global_start = pos_S - take
            global_end = pos_S - 1
            if global_start < 0:
                possible = False
            else:
                sum_global_part = prefix[global_end + 1] - prefix[global_start]
                overlap_in_subset = (lo_i <= global_end) and (hi_i > global_start)
                if overlap_in_subset:
                    sum_B_part = sum_global_part - original_A_i
                else:
                    sum_B_part = sum_global_part
                sum_d = take * (max_S + 1) - sum_B_part
                R = (K - sumA + original_A_i) - max_S
                if sum_d > R:
                    possible = True
        
        if not possible:
            res.append(-1)
            continue
        
        # Perform binary search
        low = original_A_i
        high = max_S
        answer_S = None
        while low <= high:
            mid = (low + high) // 2
            pos_S = bisect.bisect_right(sorted_A, mid)
            global_count = N - pos_S
            if original_A_i > mid:
                z = global_count - 1
            else:
                z = global_count
            t_global = pos_S
            if original_A_i <= mid:
                t = t_global - 1
            else:
                t = t_global
            
            if z >= M:
                low = mid + 1
                continue
            
            take = M - z
            global_start = pos_S - take
            global_end = pos_S - 1
            if global_start < 0:
                low = mid + 1
                continue
            
            sum_global_part = prefix[global_end + 1] - prefix[global_start]
            overlap_in_subset = (lo_i <= global_end) and (hi_i > global_start)
            if overlap_in_subset:
                sum_B_part = sum_global_part - original_A_i
            else:
                sum_B_part = sum_global_part
            
            sum_d = take * (mid + 1) - sum_B_part
            R = (K - sumA + original_A_i) - mid
            if sum_d > R:
                answer_S = mid
                high = mid - 1
            else:
                low = mid + 1
        
        if answer_S is not None:
            res.append(answer_S - original_A_i)
        else:
            res.append(-1)
    
    print(' '.join(map(str, res)))

if __name__ == '__main__':
    main()