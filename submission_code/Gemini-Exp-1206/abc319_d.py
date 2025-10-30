def solve():
    n, m = map(int, input().split())
    l = list(map(int, input().split()))

    def check(w):
        lines = 1
        current_width = 0
        for i in range(n):
            if l[i] > w:
                return False
            if current_width == 0:
                current_width = l[i]
            elif current_width + 1 + l[i] <= w:
                current_width += 1 + l[i]
            else:
                lines += 1
                current_width = l[i]
        return lines <= m

    left = 1
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