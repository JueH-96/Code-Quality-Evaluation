def solve():
    n, q = map(int, input().split())
    colors = list(range(1, n + 1))

    def get_reachable(x):
        reachable = []
        q = [x - 1]
        visited = [False] * n
        visited[x - 1] = True
        reachable.append(x - 1)

        while q:
            curr = q.pop(0)
            curr_color = colors[curr]

            # Check left neighbor
            if curr > 0 and not visited[curr - 1] and colors[curr - 1] == curr_color:
                q.append(curr - 1)
                visited[curr - 1] = True
                reachable.append(curr - 1)
            
            # Check right neighbor
            if curr < n - 1 and not visited[curr + 1] and colors[curr + 1] == curr_color:
                q.append(curr + 1)
                visited[curr + 1] = True
                reachable.append(curr + 1)
        
        return reachable

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            x, c = query[1], query[2]
            reachable = get_reachable(x)
            for i in reachable:
                colors[i] = c
        else:
            c = query[1]
            count = 0
            for color in colors:
                if color == c:
                    count += 1
            print(count)

solve()