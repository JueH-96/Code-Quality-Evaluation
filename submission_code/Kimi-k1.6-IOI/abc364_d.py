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
        b = int(input[ptr])
        k = int(input[ptr+1])
        ptr += 2
        low = 0
        high = 10**18
        while low < high:
            mid = (low + high) // 2
            left = b - mid
            right = b + mid
            cnt = bisect.bisect_right(a, right) - bisect.bisect_left(a, left)
            if cnt < k:
                low = mid + 1
            else:
                high = mid
        print(low)

if __name__ == "__main__":
    main()