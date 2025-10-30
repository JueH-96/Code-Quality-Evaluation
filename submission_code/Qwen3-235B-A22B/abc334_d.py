import sys
import bisect

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    R = list(map(int, input[ptr:ptr + N]))
    ptr += N
    queries = list(map(int, input[ptr:ptr + Q]))
    
    R.sort()
    
    prefix = [0]
    for r in R:
        prefix.append(prefix[-1] + r)
    
    res = []
    for x in queries:
        k = bisect.bisect_right(prefix, x) - 1
        res.append(str(k))
    
    print('
'.join(res))

if __name__ == '__main__':
    main()