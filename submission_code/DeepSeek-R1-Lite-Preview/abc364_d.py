import sys
import bisect

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    A = []
    for _ in range(N):
        A.append(int(data[ptr]))
        ptr += 1
    A_sorted = sorted(A)
    results = []
    for _ in range(Q):
        Bj = int(data[ptr])
        ptr += 1
        kj = int(data[ptr])
        ptr += 1
        # Binary search for the smallest D where at least kj points have |Bj - Ai| <= D
        left = 0
        right = 2 * 10**8  # Maximum possible distance
        while left < right:
            mid = (left + right) // 2
            # Find the number of Ai in [Bj - mid, Bj + mid]
            left_idx = bisect.bisect_left(A_sorted, Bj - mid)
            right_idx = bisect.bisect_right(A_sorted, Bj + mid)
            count = right_idx - left_idx
            if count >= kj:
                right = mid
            else:
                left = mid + 1
        results.append(str(left))
    print('
'.join(results))

if __name__ == '__main__':
    main()