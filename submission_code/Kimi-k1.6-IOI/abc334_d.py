import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx +=2
    R = list(map(int, data[idx:idx+N]))
    idx +=N
    R.sort()
    prefix_sum = [0]
    current = 0
    for r in R:
        current += r
        prefix_sum.append(current)
    results = []
    for _ in range(Q):
        X = int(data[idx])
        idx +=1
        res = bisect.bisect_right(prefix_sum, X) -1
        results.append(res)
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()