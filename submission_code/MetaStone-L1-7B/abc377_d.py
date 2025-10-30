import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    m = int(data[1])
    
    intervals = []
    index = 2
    for _ in range(n):
        l = int(data[index])
        r = int(data[index + 1])
        intervals.append((r, l))
        index += 2
    
    intervals.sort()
    R_list = [x[0] for x in intervals]
    L_list = [x[1] for x in intervals]
    
    prefix_max = [0] * len(intervals)
    prefix_max[0] = L_list[0]
    for i in range(1, len(intervals)):
        prefix_max[i] = max(prefix_max[i - 1], L_list[i])
    
    sum_forbidden = 0
    for r in range(1, m + 1):
        idx = bisect.bisect_right(R_list, r) - 1
        if idx >= 0:
            sum_forbidden += prefix_max[idx]
        else:
            sum_forbidden += 0
    
    total_pairs = m * (m + 1) // 2
    answer = total_pairs - sum_forbidden
    print(answer)

if __name__ == '__main__':
    main()