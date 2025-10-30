for _ in range(int(input())):
    input()
    res = 0
    l = 0
    p = list(map(int,input().split()))
    for r,c in enumerate(p):
        n = c > r + 1
        if n != p[r-1] > r : l += n
        if r + 1 == c : res = max(res,l);l = 0
    print(res)