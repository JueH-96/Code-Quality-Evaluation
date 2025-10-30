def solve():
    n, s = map(int, input().split())
    a = list(map(int, input().split()))

    for start in range(n):
        current_sum = 0
        for end in range(start, start + 2 * n):
            current_sum += a[end % n]
            if current_sum == s:
                print("Yes")
                return

    print("No")

solve()