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
    a = list(map(int, data[idx:idx+N]))
    idx += N
    a.sort()
    
    for _ in range(Q):
        b = int(data[idx])
        idx += 1
        k = int(data[idx])
        idx += 1
        
        max_dist_left = b - a[0]
        max_dist_right = a[-1] - b
        high = max(max_dist_left, max_dist_right)
        low = 0
        ans = high
        
        while low <= high:
            mid = (low + high) // 2
            current_low = b - mid
            current_high = b + mid
            left = bisect.bisect_left(a, current_low)
            right = bisect.bisect_right(a, current_high)
            count = right - left
            if count >= k:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        print(ans)

if __name__ == "__main__":
    main()