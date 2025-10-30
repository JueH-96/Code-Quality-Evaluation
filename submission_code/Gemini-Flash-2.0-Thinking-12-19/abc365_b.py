def solve():
    n = int(input())
    a = list(map(int, input().split()))

    sorted_a = sorted(a, reverse=True)
    second_largest = sorted_a[1]

    for i in range(n):
        if a[i] == second_largest:
            print(i + 1)
            return

solve()