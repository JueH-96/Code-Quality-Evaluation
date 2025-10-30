import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    intervals = []
    index = 1
    for _ in range(n):
        l = int(data[index])
        r = int(data[index + 1])
        intervals.append((l, r))
        index += 2
    
    intervals.sort()
    L = [x[0] for x in intervals]
    
    count = 0
    for i in range(n):
        r_i = intervals[i][1]
        idx = bisect.bisect_right(L, r_i) - 1
        if idx > i:
            count += (idx - i)
    
    print(count)

if __name__ == "__main__":
    main()