def solve():
    n = int(input())
    a = list(map(int, input().split()))

    while True:
        inserted = False
        for i in range(len(a) - 1):
            diff = abs(a[i] - a[i+1])
            if diff != 1:
                if a[i] < a[i+1]:
                    to_insert = list(range(a[i] + 1, a[i+1]))
                else:
                    to_insert = list(range(a[i] - 1, a[i+1], -1))
                a = a[:i+1] + to_insert + a[i+1:]
                inserted = True
                break
        if not inserted:
            break

    print(*a)

solve()