def solve():
    n, m = map(int, input().split())
    s = input()
    t = input()

    q = [["#" * n, 0]]
    visited = {"#" * n}

    while q:
        curr_x, steps = q.pop(0)
        if curr_x == s:
            print("Yes")
            return

        for i in range(n - m + 1):
            next_x = list(curr_x)
            next_x[i:i+m] = list(t)
            next_x = "".join(next_x)
            if next_x not in visited:
                visited.add(next_x)
                q.append([next_x, steps + 1])
    
    print("No")

solve()