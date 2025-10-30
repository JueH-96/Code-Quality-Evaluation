import bisect
import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1

    R = list(map(int, data[ptr:ptr+N]))
    ptr += N

    R.sort()
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + R[i]
    
    output = []
    for _ in range(Q):
        X = int(data[ptr])
        ptr += 1
        k = bisect.bisect_right(prefix, X) - 1
        output.append(str(k))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()