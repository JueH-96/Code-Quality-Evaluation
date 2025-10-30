read = lambda: [*map(int, input().split())]
n, = read()
a = read()
a += a[1:-1]
l = 0
r = len(a)
mx = 0
while l <= r:
    m = (l+r)//2
    mx = max(mx, m)
    p1 = k = cnt = flag = 0
    for p2, v in enumerate(a):
        if p2 - p1 < m:
            if v > m-1 or flag and v < m-1:
                break
            cnt += m - 1 - v
            k = (k + 1) % 2
            if k == 0 and v < m:
                flag = 1
        else:
            if flag and a[p1] > m-1:
                break
            cnt -= m - 1 - a[p1]
            p1 += 1
            k = (k + 1) % 2
    else:
        l = m + 1
    if cnt <= mx:
        l = m + 1
    else:
        r = m - 1
print(l//2)