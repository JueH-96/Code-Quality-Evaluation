import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    R = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    R.sort()
    prefix = [0]
    for num in R:
        prefix.append(prefix[-1] + num)
    
    for _ in range(Q):
        X = int(input[ptr])
        ptr += 1
        res = bisect.bisect_right(prefix, X) - 1
        print(res)

if __name__ == '__main__':
    main()