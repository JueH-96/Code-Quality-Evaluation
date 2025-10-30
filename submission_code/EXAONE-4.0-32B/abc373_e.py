import bisect

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:3+n]))
    
    total = sum(A)
    R = K - total
    
    B = sorted(A)
    P = [0] * (n+1)
    for i in range(1, n+1):
        P[i] = P[i-1] + B[i-1]
        
    results = []
    for a in A:
        low, high = 0, R
        ans = -1
        while low <= high:
            mid = (low + high) // 2
            v = a + mid
            idx = bisect.bisect_right(B, v)
            base_above = n - idx
            
            if base_above >= M:
                low = mid + 1
            else:
                k0 = M - base_above
                non_above_size = idx - 1
                
                if k0 > non_above_size:
                    ans = mid
                    high = mid - 1
                else:
                    start_index = idx - (k0 + 1)
                    total_block = P[idx] - P[start_index]
                    block_min = B[start_index]
                    
                    if a >= block_min:
                        S = total_block - a
                    else:
                        S = total_block - block_min
                    
                    if k0 * (a + mid + 1) - S > R - mid:
                        ans = mid
                        high = mid - 1
                    else:
                        low = mid + 1
        results.append(str(ans))
    
    print(" ".join(results))

if __name__ == "__main__":
    main()