import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    m = int(data[1])
    a = list(map(int, data[2:2+n]))
    a.sort()
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + a[i]
    total = prefix[-1]
    if total <= m:
        print("infinite")
    else:
        low = 0
        high = a[-1]
        max_x = 0
        while low <= high:
            mid = (low + high) // 2
            idx = bisect.bisect_right(a, mid)
            current = prefix[idx] + mid * (n - idx)
            if current <= m:
                max_x = mid
                low = mid + 1
            else:
                high = mid - 1
        print(max_x)

if __name__ == "__main__":
    main()