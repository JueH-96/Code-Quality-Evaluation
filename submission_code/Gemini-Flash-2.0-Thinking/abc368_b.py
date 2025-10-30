def solve():
    n = int(input())
    a = list(map(int, input().split()))

    operations = 0
    while True:
        positive_count = sum(1 for x in a if x > 0)
        if positive_count <= 1:
            break

        a.sort(reverse=True)

        if len(a) >= 1 and a[0] > 0:
            a[0] -= 1
        if len(a) >= 2 and a[1] > 0:
            a[1] -= 1

        operations += 1

    print(operations)

solve()