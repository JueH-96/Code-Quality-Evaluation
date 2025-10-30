import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    R = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    R.sort()
    prefix = [0] * (N + 1)
    for i in range(1, N+1):
        prefix[i] = prefix[i-1] + R[i-1]
    
    for _ in range(Q):
        X = int(input[ptr])
        ptr += 1
        max_sleighs = bisect.bisect_right(prefix, X) - 1
        print(max_sleighs)

if __name__ == "__main__":
    main()