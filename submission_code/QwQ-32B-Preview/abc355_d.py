import sys
import bisect

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    intervals = []
    for i in range(N):
        l = int(data[1 + 2 * i])
        r = int(data[1 + 2 * i + 1])
        intervals.append((l, r))
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    sorted_l = [l for l, r in sorted_intervals]
    count = 0
    for i in range(N-1):
        r_i = sorted_intervals[i][1]
        j = bisect.bisect_right(sorted_l, r_i, lo=i+1)
        count += j - (i+1)
    print(count)

if __name__ == '__main__':
    main()