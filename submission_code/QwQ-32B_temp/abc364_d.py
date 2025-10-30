import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N, Q = int(input[ptr]), int(input[ptr+1])
    ptr += 2
    a = list(map(int, input[ptr:ptr+N]))
    ptr += N
    a.sort()
    for _ in range(Q):
        b = int(input[ptr])
        k = int(input[ptr+1])
        ptr += 2
        leftmost = a[0]
        rightmost = a[-1]
        d1 = abs(leftmost - b)
        d2 = abs(rightmost - b)
        max_dist = max(d1, d2)
        low = 0
        high = max_dist
        ans = max_dist
        while low <= high:
            mid = (low + high) // 2
            L = b - mid
            R = b + mid
            left_idx = bisect.bisect_left(a, L)
            right_idx = bisect.bisect_right(a, R)
            cnt = right_idx - left_idx
            if cnt >= k:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        print(ans)

if __name__ == "__main__":
    main()