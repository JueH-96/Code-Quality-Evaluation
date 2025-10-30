import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    t = int(data[1])
    s = data[2]
    x = list(map(int, data[3:3+n]))
    
    # Create list of (X_i, S_i) sorted by X_i
    ants = sorted(zip(x, s), key=lambda a: a[0])
    
    zeros = []
    ones = []
    
    for xi, si in ants:
        if si == '0':
            zeros.append(xi)
        else:
            ones.append(xi)
    
    max_diff = 2 * t
    count = 0
    
    for x_val in ones:
        left = bisect.bisect_right(zeros, x_val)
        right = bisect.bisect_right(zeros, x_val + max_diff)
        count += right - left
    
    print(count)

if __name__ == "__main__":
    main()