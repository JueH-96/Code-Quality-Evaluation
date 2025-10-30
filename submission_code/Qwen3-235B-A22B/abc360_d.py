import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    T = int(input[idx])
    idx += 1
    s = input[idx]
    idx += 1
    x = list(map(int, input[idx:idx+n]))
    
    R = []
    L = []
    for i in range(n):
        if s[i] == '1':
            R.append(x[i])
        else:
            L.append(x[i])
    
    R.sort()
    L.sort()
    
    ans = 0
    dx_max = 2 * T
    
    for xr in R:
        left = xr + 1
        right = xr + dx_max
        # Count L elements >= left and <= right
        l_pos = bisect.bisect_left(L, left)
        r_pos = bisect.bisect_right(L, right)
        ans += r_pos - l_pos
    
    print(ans)

if __name__ == '__main__':
    main()