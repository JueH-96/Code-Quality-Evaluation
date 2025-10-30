def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    
    for _ in range(q):
        i, x = map(int, input().split())
        a[i - 1] = x
        
        seen = set()
        for val in a:
            seen.add(val)
        
        mex = 0
        while mex in seen:
            mex += 1
        print(mex)

solve()