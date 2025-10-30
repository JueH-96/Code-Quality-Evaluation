import sys
import bisect

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    X = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    P = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    # Compute prefix sums
    prefix = [0] * (N +1)
    for i in range(N):
        prefix[i+1] = prefix[i] + P[i]
    Q = int(input[ptr])
    ptr +=1
    for _ in range(Q):
        L = int(input[ptr])
        R = int(input[ptr+1])
        ptr +=2
        l = bisect.bisect_left(X, L)
        r = bisect.bisect_right(X, R) -1
        if l > r:
            print(0)
        else:
            print(prefix[r+1] - prefix[l])
            
if __name__ == '__main__':
    main()