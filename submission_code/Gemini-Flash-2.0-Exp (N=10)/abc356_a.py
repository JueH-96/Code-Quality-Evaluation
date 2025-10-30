def solve():
    n, l, r = map(int, input().split())
    a = list(range(1, n + 1))
    l -= 1
    sub_array = a[l:r]
    sub_array.reverse()
    a[l:r] = sub_array
    print(*a)

solve()