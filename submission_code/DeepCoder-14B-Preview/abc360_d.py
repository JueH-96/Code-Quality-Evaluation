import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    t = int(data[1])
    s = data[2]
    x_list = list(map(int, data[3:3+n]))
    
    R = []
    L = []
    for i in range(n):
        if s[i] == '1':
            R.append(x_list[i])
        else:
            L.append(x_list[i])
    
    R_sorted = sorted(R)
    L_sorted = sorted(L)
    
    count = 0
    
    # Process R to L pairs
    for x in R_sorted:
        low = x
        high = x + 2 * (t + 0.1)
        left = bisect.bisect_right(L_sorted, low)
        right = bisect.bisect_right(L_sorted, high)
        count += (right - left)
    
    print(count)

if __name__ == '__main__':
    main()