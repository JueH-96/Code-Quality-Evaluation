import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    a = list(map(int, input[ptr:ptr+N]))
    ptr += N
    a.sort()
    for _ in range(Q):
        x = int(input[ptr])
        ptr += 1
        k = int(input[ptr])
        ptr += 1
        low = 0
        # Compute the maximum initial distance for binary search
        a0 = a[0]
        aN = a[-1]
        max_distance = max(abs(a0 - x), abs(aN - x))
        high = max_distance
        while low < high:
            mid = (low + high) // 2
            left = bisect.bisect_left(a, x - mid)
            right = bisect.bisect_right(a, x + mid)
            count = right - left
            if count >= k:
                high = mid
            else:
                low = mid + 1
        print(low)

if __name__ == "__main__":
    main()