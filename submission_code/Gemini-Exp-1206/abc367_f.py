def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for _ in range(q):
        l, r, L, R = map(int, input().split())
        l -= 1
        L -= 1
        
        if r - l != R - L:
            print("No")
            continue

        sub_a = a[l:r]
        sub_b = b[L:R]
        
        if sorted(sub_a) == sorted(sub_b):
            print("Yes")
        else:
            print("No")

solve()