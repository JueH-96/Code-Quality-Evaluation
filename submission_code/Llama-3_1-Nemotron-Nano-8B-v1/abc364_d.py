import bisect
import sys

def main():
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
        x = b
        low = 0
        max_dist = max(abs(a[0] - x), abs(a[-1] - x))
        high = max_dist
        ans = high
        while low <= high:
            mid = (low + high) // 2
            left = bisect.bisect_left(a, x - mid)
            right_end = bisect.bisect_right(a, x + mid)
            right = right_end - 1
            count = right - left + 1 if left <= right else 0
            if count >= k:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        print(ans)

if __name__ == '__main__':
    main()