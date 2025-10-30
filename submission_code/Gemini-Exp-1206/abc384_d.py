def solve():
    n, s = map(int, input().split())
    a = list(map(int, input().split()))

    for i in range(n):
        current_sum = 0
        for j in range(i, i + 2 * n):
            current_sum += a[j % n]
            if current_sum == s:
                print("Yes")
                return
            elif current_sum > s:
                break

    print("No")

solve()