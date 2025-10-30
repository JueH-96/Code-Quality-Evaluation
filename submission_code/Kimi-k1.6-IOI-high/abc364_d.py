import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx +=2
    a = list(map(int, data[idx:idx+N]))
    idx +=N
    a.sort()
    for _ in range(Q):
        b = int(data[idx])
        k = int(data[idx+1])
        idx +=2
        high = max(abs(b - a[0]), abs(b - a[-1]))
        left = 0
        right = high
        while left < right:
            mid = (left + right) // 2
            lower = b - mid
            upper = b + mid
            l = bisect.bisect_left(a, lower)
            r = bisect.bisect_right(a, upper)
            cnt = r - l
            if cnt >= k:
                right = mid
            else:
                left = mid + 1
        print(left)

if __name__ == '__main__':
    main()