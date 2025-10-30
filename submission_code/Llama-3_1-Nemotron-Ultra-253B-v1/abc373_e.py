import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:3+N]))
    sum_A = sum(A)
    
    B = sorted(A, reverse=True)
    pre_sum = [0] * (N + 1)
    for i in range(N):
        pre_sum[i+1] = pre_sum[i] + B[i]
    
    # Precompute for each element in A, the left and right indices in B
    left = []
    right = []
    for a in A:
        l = bisect.bisect_left(B, a, 0, N, key=lambda x: -x)
        r = bisect.bisect_right(B, a, 0, N, key=lambda x: -x)
        left.append(l)
        right.append(r)
    
    result = []
    for i in range(N):
        a_i = A[i]
        L = left[i]
        R = right[i]
        
        low = 0
        high = K - sum_A
        best = -1
        
        # Check if already satisfied with X=0
        if high < 0:
            result.append(-1)
            continue
        
        while low <= high:
            mid = (low + high) // 2
            X = mid
            T_i = a_i + X
            S = bisect.bisect_left(B, T_i, 0, N, key=lambda x: -x)
            R_remaining = K - sum_A - X
            if R_remaining < 0:
                low = mid + 1
                continue
            
            t_max = (N - S) - 1
            if t_max < 0:
                total = S
                if total <= M-1:
                    best = mid
                    high = mid - 1
                else:
                    low = mid + 1
                continue
            
            # Binary search for t in [0, t_max]
            t_low = 0
            t_high = t_max
            t_best = 0
            while t_low <= t_high:
                t_mid = (t_low + t_high) // 2
                end = S + t_mid
                if end > N:
                    t_high = t_mid - 1
                    continue
                
                # Calculate the number of elements equal to a_i in B[S:end]
                overlap_start = max(S, L)
                overlap_end = min(end, R)
                count = max(0, overlap_end - overlap_start)
                
                sum_t = (pre_sum[end] - pre_sum[S]) - count * a_i
                sum_req = t_mid * (T_i + 1) - sum_t
                
                if sum_req <= R_remaining:
                    t_best = t_mid
                    t_low = t_mid + 1
                else:
                    t_high = t_mid - 1
            
            total = S + t_best
            if total <= M-1:
                best = mid
                high = mid - 1
            else:
                low = mid + 1
        
        result.append(best if best != -1 else -1)
    
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()