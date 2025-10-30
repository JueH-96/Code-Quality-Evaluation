for _ in range(int(input())):
    n = int(input())
    a = list(map(int, input().split()))
    if 0 in a:
        print(prod(a) // a[a.index(0)])
    elif max(a) == 9:
        print(prod(a) * 2)
    else:
        print(prod(a) // a[a.index(max(a))] * (max(a) + 1))