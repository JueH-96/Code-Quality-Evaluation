def solve():
    n, q = map(int, input().split())
    colors = list(range(1, n + 1))

    def get_reachable(start_index):
        reachable = set()
        stack = [start_index]
        start_color = colors[start_index]
        
        while stack:
            curr_index = stack.pop()
            if curr_index < 0 or curr_index >= n or curr_index in reachable or colors[curr_index] != start_color:
                continue
            
            reachable.add(curr_index)
            stack.append(curr_index - 1)
            stack.append(curr_index + 1)
        
        return reachable

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            x, c = query[1] - 1, query[2]
            reachable_cells = get_reachable(x)
            for cell_index in reachable_cells:
                colors[cell_index] = c
        else:
            c = query[1]
            count = colors.count(c)
            print(count)

solve()