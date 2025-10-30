import sys
import bisect

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    R = []
    for _ in range(N):
        R.append(int(data[ptr]))
        ptr += 1
    R.sort()
    
    # Compute prefix sums
    P = []
    current_sum = 0
    for r in R:
        current_sum += r
        P.append(current_sum)
    
    # Process queries
    for _ in range(Q):
        X = int(data[ptr])
        ptr += 1
        ans = bisect.bisect_right(P, X)
        print(ans)

if __name__ == '__main__':
    main()