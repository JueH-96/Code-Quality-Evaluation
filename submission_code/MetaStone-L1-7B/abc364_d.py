import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    
    A = list(map(int, data[idx:idx+N]))
    idx += N
    A.sort()
    
    results = []
    for _ in range(Q):
        b = int(data[idx])
        idx += 1
        k = int(data[idx])
        idx += 1
        
        low = 0
        high = 2 * 10**18
        ans = high
        
        while low <= high:
            mid = (low + high) // 2
            left = bisect.bisect_left(A, b - mid)
            right = bisect.bisect_right(A, b + mid)
            cnt = right - left
            
            if cnt >= k:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        results.append(ans)
    
    print('
'.join(map(str, results)))

if __name__ == '__main__':
    main()