def solve():
    n, m = map(int, input().split())
    l = list(map(int, input().split()))

    def check(width):
        lines = 1
        current_width = 0
        for word_width in l:
            if current_width == 0:
                current_width = word_width
            elif current_width + 1 + word_width <= width:
                current_width += 1 + word_width
            else:
                lines += 1
                current_width = word_width
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