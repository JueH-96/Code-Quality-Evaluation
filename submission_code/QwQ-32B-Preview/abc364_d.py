import sys
import bisect

def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    A = list(map(int, data[idx:idx+N]))
    idx += N
    A.sort()
    for _ in range(Q):
        bj = int(data[idx])
        kj = int(data[idx+1])
        idx += 2
        low = 0
        high = 2 * 10**8  # Maximum possible distance
        while low < high:
            mid = (low + high) // 2
            left = bisect.bisect_left(A, bj - mid)
            right = bisect.bisect_right(A, bj + mid, lo=left)
            count = right - left
            if count >= kj:
                high = mid
            else:
                low = mid + 1
        print(low)

if __name__ == "__main__":
    main()