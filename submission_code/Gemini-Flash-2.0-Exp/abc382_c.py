def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for sushi_deliciousness in b:
        eaten = False
        for i in range(n):
            if sushi_deliciousness >= a[i]:
                print(i + 1)
                eaten = True
                break
        if not eaten:
            print(-1)

solve()