def solve():
    n, q = map(int, input().split())
    colors = list(range(1, n + 1))
    color_counts = [0] * (n + 1)
    for color in colors:
        color_counts[color] += 1

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            x, c = query[1], query[2]
            x -= 1
            initial_color = colors[x]
            if initial_color == c:
                continue

            left = x
            while left > 0 and colors[left - 1] == initial_color:
                left -= 1
            right = x
            while right < n - 1 and colors[right + 1] == initial_color:
                right += 1

            count = right - left + 1
            color_counts[initial_color] -= count
            color_counts[c] += count
            for i in range(left, right + 1):
                colors[i] = c
        elif query[0] == 2:
            c = query[1]
            print(color_counts[c])

solve()