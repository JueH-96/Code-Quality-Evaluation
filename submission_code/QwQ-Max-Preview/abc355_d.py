import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    intervals = []
    idx = 1
    for _ in range(N):
        l = int(data[idx])
        r = int(data[idx + 1])
        intervals.append((l, r))
        idx += 2
    intervals.sort()
    sorted_l = [x[0] for x in intervals]
    total = N * (N - 1) // 2
    sum_non_overlap = 0
    for i in range(N):
        r_i = intervals[i][1]
        k = bisect.bisect_right(sorted_l, r_i)
        start = max(k, i + 1)
        sum_non_overlap += (N - start)
    answer = total - sum_non_overlap
    print(answer)

if __name__ == "__main__":
    main()