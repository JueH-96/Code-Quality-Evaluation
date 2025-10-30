def solve():
    n = int(input())
    a = list(map(int, input().split()))

    b = []
    for i in range(n):
        last_index = -1
        for j in range(i):
            if a[i] == a[j]:
                last_index = j + 1
        b.append(last_index)

    print(*(b))

solve()