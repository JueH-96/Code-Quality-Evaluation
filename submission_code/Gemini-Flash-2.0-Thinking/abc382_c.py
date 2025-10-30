def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for sushi_deliciousness in b:
        eater = -1
        for i in range(n):
            if sushi_deliciousness >= a[i]:
                eater = i + 1
                break
        print(eater)

solve()