def solve():
    n = int(input())
    a = list(map(int, input().split()))

    swaps = []
    for i in range(n):
        if a[i] != i + 1:
            for j in range(i + 1, n):
                if a[j] == i + 1:
                    a[i], a[j] = a[j], a[i]
                    swaps.append((i + 1, j + 1))
                    break

    print(len(swaps))
    for swap in swaps:
        print(swap[0], swap[1])

solve()