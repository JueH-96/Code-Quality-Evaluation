import bisect

def main():
    import sys
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
        b = int(input[ptr])
        ptr += 1
        k = int(input[ptr])
        ptr += 1
        # Compute initial_high
        if N == 0:
            print(0)
            continue
        max_left = abs(b - a[0])
        max_right = abs(b - a[-1])
        initial_high = max(max_left, max_right)
        low = 0
        high = initial_high
        while low < high:
            mid = (low + high) // 2
            left_val = b - mid
            right_val = b + mid
            left_pos = bisect.bisect_left(a, left_val)
            right_pos = bisect.bisect_right(a, right_val)
            count = right_pos - left_pos
            if count >= k:
                high = mid
            else:
                low = mid + 1
        print(low)

if __name__ == '__main__':
    main()