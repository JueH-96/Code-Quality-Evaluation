def solve():
    n = int(input())
    a = list(map(int, input().split()))

    sorted_a = sorted(a)
    largest = sorted_a[-1]

    for i in range(n - 2, -1, -1):
        if sorted_a[i] != largest:
            print(sorted_a[i])
            return

solve()