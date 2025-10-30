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
    l_list = [l for l, r in intervals]
    ans = 0
    for i in range(n):
        r_i = intervals[i][1]
        cnt = bisect.bisect_right(l_list, r_i, i + 1, n) - (i + 1)
        ans += cnt
    print(ans)

if __name__ == "__main__":
    main()