import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    m = int(data[idx])
    idx += 1
    
    intervals = []
    for _ in range(n):
        l = int(data[idx])
        idx += 1
        r = int(data[idx])
        idx += 1
        intervals.append((l, r))
    
    # Sort intervals in decreasing order of L_i
    intervals.sort(key=lambda x: -x[0])
    
    L_list = [x[0] for x in intervals]
    R_list = [x[1] for x in intervals]
    
    # Create the neg_L list for binary search
    neg_L = [-x for x in L_list]
    
    # Compute the prefix_min array
    prefix_min = []
    if len(R_list) > 0:
        prefix_min = [R_list[0]]
        for i in range(1, len(R_list)):
            prefix_min.append(min(prefix_min[-1], R_list[i]))
    
    forbidden = 0
    for l in range(1, m + 1):
        # Find the number of intervals with L_i >= l
        target = -l
        count = bisect.bisect_right(neg_L, target)
        if count == 0:
            continue
        else:
            if count - 1 >= len(prefix_min):
                minimal_R = float('inf')
            else:
                minimal_R = prefix_min[count - 1]
            if minimal_R > m:
                continue
            else:
                forbidden += (m - minimal_R + 1)
    
    total = m * (m + 1) // 2
    valid = total - forbidden
    print(valid)

if __name__ == '__main__':
    main()