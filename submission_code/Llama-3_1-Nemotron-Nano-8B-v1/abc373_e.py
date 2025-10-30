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
    S = sorted(A)
    prefix_S = [0] * (N + 1)
    for i in range(N):
        prefix_S[i+1] = prefix_S[i] + S[i]
    
    results = []
    
    for i in range(N):
        a_i = A[i]
        left = bisect.bisect_left(S, a_i)
        right = bisect.bisect_right(S, a_i)
        count_a = right - left
        
        low = 0
        high = K - sum_A
        best_X = -1
        
        while low <= high:
            mid = (low + high) // 2
            X = mid
            current_i = a_i + X
            remaining = K - sum_A - X
            
            pos = bisect.bisect_right(S, current_i)
            count_le_S = pos
            count_le_Sprime = count_le_S - (1 if a_i <= current_i else 0)
            C_initial = (N-1) - count_le_Sprime
            
            sum_S_le = prefix_S[pos]
            if a_i <= current_i:
                sum_S_le -= a_i
            
            sum_required = (current_i + 1) * count_le_Sprime - sum_S_le
            
            if sum_required <= remaining:
                count = C_initial + count_le_Sprime
                if count < M:
                    best_X = X
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                low = mid + 1
        
        if best_X != -1:
            results.append(best_X)
        else:
            results.append(-1)
    
    print(' '.join(map(str, results)))

if __name__ == "__main__":
    main()