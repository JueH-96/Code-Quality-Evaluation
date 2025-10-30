def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for _ in range(q):
        l, r, L, R = map(int, input().split())
        sub_a = a[l-1:r]
        sub_b = b[L-1:R]
        
        if len(sub_a) != len(sub_b):
            print("No")
            continue
            
        counts_a = {}
        for x in sub_a:
            counts_a[x] = counts_a.get(x, 0) + 1
        
        counts_b = {}
        for x in sub_b:
            counts_b[x] = counts_b.get(x, 0) + 1
            
        if counts_a == counts_b:
            print("Yes")
        else:
            print("No")

solve()