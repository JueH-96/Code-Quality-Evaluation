def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    intervals = []
    for i in range(N):
        L = int(data[2 + 2*i])
        R = int(data[3 + 2*i])
        intervals.append((R, L))
    
    intervals.sort()  # Sort by R_i
    
    max_L = 0
    index = 0
    total = 0
    
    for r in range(1, M+1):
        while index < N and intervals[index][0] <= r:
            max_L = max(max_L, intervals[index][1])
            index += 1
        valid_l = r - max_L
        if valid_l > 0:
            total += valid_l
        else:
            total += 0  # No valid l for this r
    
    print(total)

if __name__ == '__main__':
    main()