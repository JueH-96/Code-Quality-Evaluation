def solve():
    n, m = map(int, input().split())
    l = list(map(int, input().split()))

    def check(width):
        lines = 1
        current_width = 0
        for i in range(n):
            if i == 0:
                current_width = l[i]
            else:
                if current_width + 1 + l[i] <= width:
                    current_width += 1 + l[i]
                else:
                    lines += 1
                    current_width = l[i]
        return lines <= m

    left = max(l)
    right = sum(l) + n - 1
    ans = right

    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            ans = mid
            right = mid - 1
        else:
            left = mid + 1

    print(ans)

solve()