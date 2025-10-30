import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    intervals = []
    rs = []
    ls = []
    index = 1
    for _ in range(n):
        l = int(data[index])
        r = int(data[index + 1])
        intervals.append((l, r))
        rs.append(r)
        ls.append(l)
        index += 2
    
    rs_sorted = sorted(rs)
    ls_sorted = sorted(ls)
    
    total = n * (n - 1) // 2
    non_overlapping = 0
    
    for l, r in intervals:
        left = bisect.bisect_left(rs_sorted, l)
        right_pos = bisect.bisect_right(ls_sorted, r)
        right = len(ls_sorted) - right_pos
        non_overlapping += left + right
    
    non_overlapping //= 2
    print(total - non_overlapping)

if __name__ == "__main__":
    main()