def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))

    for _ in range(q):
        i, x = map(int, input().split())
        a[i-1] = x
        
        s = set(a)
        mex = 0
        while mex in s:
            mex += 1
        print(mex)

solve()