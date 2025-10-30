def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))

    for i in range(1, n + 1):
        left, right = 0, m - 1
        ans = -1

        while left <= right:
            mid = (left + right) // 2
            if a[mid] >= i:
                ans = a[mid]
                right = mid - 1
            else:
                left = mid + 1

        print(ans - i)

solve()