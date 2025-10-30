import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    m = int(data[idx+1])
    idx += 2
    
    intervals = []
    for _ in range(n):
        l = int(data[idx])
        r = int(data[idx+1])
        intervals.append((l, r))
        idx += 2
    
    # Sort intervals by R in ascending order
    intervals.sort(key=lambda x: x[1])
    sorted_R = [x[1] for x in intervals]
    sorted_L = [x[0] for x in intervals]
    
    # Build prefix max L
    prefix_max_L = []
    current_max = 0
    for i in range(n):
        current_max = max(current_max, sorted_L[i])
        prefix_max_L.append(current_max)
    
    # Compute forbidden sum
    forbidden = 0
    for r in range(1, m + 1):
        # Find all intervals with Ri <= r
        pos = bisect.bisect_right(sorted_R, r)
        j = pos - 1
        if j >= 0:
            max_L = prefix_max_L[j]
        else:
            max_L = 0
        forbidden += min(max_L, r)
    
    total = m * (m + 1) // 2
    print(total - forbidden)

if __name__ == "__main__":
    main()