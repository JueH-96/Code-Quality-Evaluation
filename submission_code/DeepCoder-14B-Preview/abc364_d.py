import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    q = int(data[idx])
    idx += 1
    
    a = list(map(int, data[idx:idx+n]))
    idx += n
    a.sort()
    
    results = []
    for _ in range(q):
        b = int(data[idx])
        idx += 1
        k = int(data[idx])
        idx += 1
        
        if not a:
            results.append(0)
            continue
        
        if b <= a[0]:
            high = a[-1] - b
        elif b >= a[-1]:
            high = b - a[0]
        else:
            high = max(b - a[0], a[-1] - b)
        
        low = 0
        while low < high:
            mid = (low + high) // 2
            L = b - mid
            R = b + mid
            left = bisect.bisect_left(a, L)
            right_idx = bisect.bisect_right(a, R)
            count = right_idx - left
            if count >= k:
                high = mid
            else:
                low = mid + 1
        results.append(low)
    
    print('
'.join(map(str, results)))

if __name__ == '__main__':
    main()