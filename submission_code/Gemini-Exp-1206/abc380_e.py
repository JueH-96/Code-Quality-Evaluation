def solve():
    n, q = map(int, input().split())
    colors = list(range(1, n + 1))

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            x, c = query[1], query[2]
            x -= 1
            target_color = colors[x]
            
            left = x
            while left > 0 and colors[left - 1] == target_color:
                left -= 1
            
            right = x
            while right < n - 1 and colors[right + 1] == target_color:
                right += 1
            
            for i in range(left, right + 1):
                colors[i] = c
        else:
            c = query[1]
            count = 0
            for color in colors:
                if color == c:
                    count += 1
            print(count)

solve()