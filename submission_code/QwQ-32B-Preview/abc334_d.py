import sys
import bisect

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    R = []
    for _ in range(N):
        R.append(int(input[ptr]))
        ptr += 1
    R.sort()
    prefix_sums = []
    current_sum = 0
    for r in R:
        current_sum += r
        prefix_sums.append(current_sum)
    for _ in range(Q):
        X = int(input[ptr])
        ptr += 1
        m = bisect.bisect_right(prefix_sums, X)
        print(m)

if __name__ == '__main__':
    main()