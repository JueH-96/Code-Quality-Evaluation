import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    X = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    P = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    prefix = [0]
    for p in P:
        prefix.append(prefix[-1] + p)
    Q = int(input[ptr])
    ptr +=1
    for _ in range(Q):
        L = int(input[ptr])
        R = int(input[ptr+1])
        ptr +=2
        left = bisect.bisect_left(X, L)
        right_idx = bisect.bisect_right(X, R) -1
        if left > right_idx:
            print(0)
        else:
            print(prefix[right_idx +1] - prefix[left])

if __name__ == "__main__":
    main()