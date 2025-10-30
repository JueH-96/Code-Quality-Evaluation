import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    t = int(data[1])
    s = data[2]
    x = list(map(int, data[3:3+n]))
    
    left = []
    right = []
    for i in range(n):
        if s[i] == '0':
            left.append(x[i])
        else:
            right.append(x[i])
    
    left.sort()
    right.sort()
    
    ans = 0
    for r in right:
        lower = r
        upper = r + 2 * t
        l = bisect.bisect_right(left, lower)
        r_idx = bisect.bisect_right(left, upper)
        ans += (r_idx - l)
    
    print(ans)

if __name__ == "__main__":
    main()