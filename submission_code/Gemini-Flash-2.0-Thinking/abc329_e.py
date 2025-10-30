def solve():
    n, m = map(int, input().split())
    s = input()
    t = input()

    initial_x = "#" * n
    queue = [initial_x]
    visited = {initial_x}

    while queue:
        current_x = queue.pop(0)
        if current_x == s:
            print("Yes")
            return

        for i in range(n - m + 1):
            can_replace = True
            for j in range(m):
                if current_x[i + j] != '#':
                    can_replace = False
                    break

            if can_replace:
                next_x_list = list(current_x)
                for j in range(m):
                    next_x_list[i + j] = t[j]
                next_x = "".join(next_x_list)
                if next_x not in visited:
                    visited.add(next_x)
                    queue.append(next_x)

    print("No")

solve()