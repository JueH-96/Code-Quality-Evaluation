def solve():
    N, M = map(int, input().split())
    L = list(map(int, input().split()))

    def can_fit(W, M, L):
        lines_used = 1
        current_line_width = 0
        for i, width in enumerate(L):
            if current_line_width == 0:
                if width > W:
                    return False
                current_line_width = width
            else:
                if current_line_width + 1 + width <= W:
                    current_line_width += 1 + width
                else:
                    lines_used += 1
                    if width > W:
                        return False
                    current_line_width = width
        return lines_used <= M

    low = max(L)
    high = sum(L) + N - 1
    ans = high + 1

    while low <= high:
        mid = (low + high) // 2
        if can_fit(mid, M, L):
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    print(ans)

solve()