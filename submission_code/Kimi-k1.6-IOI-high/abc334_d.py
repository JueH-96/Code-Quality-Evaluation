import bisect
import sys

def main():
    data = list(map(int, sys.stdin.read().split()))
    ptr = 0
    N = data[ptr]
    ptr += 1
    Q = data[ptr]
    ptr += 1
    R = data[ptr:ptr+N]
    ptr += N
    
    R.sort()
    prefix = []
    s = 0
    for num in R:
        s += num
        prefix.append(s)
    
    for _ in range(Q):
        X = data[ptr]
        ptr += 1
        ans = bisect.bisect_right(prefix, X)
        print(ans)

if __name__ == "__main__":
    main()