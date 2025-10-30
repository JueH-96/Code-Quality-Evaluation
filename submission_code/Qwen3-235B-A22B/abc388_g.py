import sys
import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    
    A = list(map(int, data[idx:idx+N]))
    idx += N
    
    Q = int(data[idx])
    idx += 1
    
    queries = []
    for _ in range(Q):
        L = int(data[idx])
        R = int(data[idx+1])
        queries.append((L, R))
        idx += 2
    
    results = []
    for L, R in queries:
        M = R - L + 1
        low = 0
        high = M // 2
        best = 0
        
        while low <= high:
            mid = (low + high) // 2
            if mid == 0:
                best = 0
                low = mid + 1
                continue
            
            # Check if all i in 0..mid-1 satisfy A[L-1 + i] * 2 <= A[R - mid + i]
            # This is O(mid) time, which is not feasible for large Q and large mid
            # Instead, use the binary search approach with the tightest condition
            # The tightest condition is for i = mid-1
            left_idx = L - 1 + (mid - 1)
            right_idx = R - mid + (mid - 1)
            
            if A[left_idx] * 2 <= A[right_idx]:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        
        results.append(str(best))
    
    print('
'.join(results))

if __name__ == '__main__':
    main()