def solve():
    n = int(input())
    a = list(map(int, input().split()))

    while True:
        insertions_made = False
        for i in range(len(a) - 1):
            if abs(a[i] - a[i+1]) != 1:
                if a[i] < a[i+1]:
                    insertion = list(range(a[i] + 1, a[i+1]))
                    a = a[:i+1] + insertion + a[i+1:]
                else:
                    insertion = list(range(a[i] - 1, a[i+1], -1))
                    a = a[:i+1] + insertion + a[i+1:]
                insertions_made = True
                break
        if not insertions_made:
            break

    print(*a)

solve()