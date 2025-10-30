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
    total_remaining = K - sum_A
    B = sorted(A)
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + B[i]
    
    result = []
    for i in range(N):
        a_i = A[i]
        pos_i = bisect.bisect_left(B, a_i)
        left = 0
        right = total_remaining
        ans = -1
        
        while left <= right:
            mid = (left + right) // 2
            X = mid
            T_plus_1 = a_i + X + 1
            R = total_remaining - X
            
            if R < 0:
                right = mid - 1
                continue
            
            pos = bisect.bisect_left(B, T_plus_1)
            count_ge = N - pos
            sum_required_all = T_plus_1 * pos - prefix_sum[pos]
            sum_required = sum_required_all - (T_plus_1 - a_i)
            count_less = pos - 1
            
            if sum_required <= R:
                total_j = count_ge + count_less
            else:
                low = 0
                high = count_less
                best_k = 0
                
                while low <= high:
                    mid_k = (low + high) // 2
                    k = mid_k
                    start = pos - k
                    if start < 0:
                        start = 0
                    sum_total = k * T_plus_1 - (prefix_sum[pos] - prefix_sum[start])
                    
                    if start <= pos_i < pos:
                        sum_without_i = sum_total - (T_plus_1 - a_i)
                    else:
                        sum_without_i = sum_total
                    
                    if sum_without_i <= R:
                        best_k = k
                        low = mid_k + 1
                    else:
                        high = mid_k - 1
                
                count_less = best_k
                total_j = count_ge + best_k
            
            if total_j < M:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        if ans == -1:
            result.append(-1)
        else:
            if ans > total_remaining:
                result.append(-1)
            else:
                result.append(ans if ans >= 0 else -1)
    
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()