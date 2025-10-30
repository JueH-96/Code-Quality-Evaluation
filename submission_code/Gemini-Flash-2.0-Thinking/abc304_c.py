import math

def solve():
    n, d = map(int, input().split())
    coords = []
    for _ in range(n):
        x, y = map(int, input().split())
        coords.append((x, y))

    infected = [False] * n
    infected[0] = True
    changed = True

    while changed:
        changed = False
        newly_infected = [False] * n
        for i in range(n):
            if infected[i]:
                for j in range(n):
                    if not infected[j]:
                        dist_sq = (coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2
                        if dist_sq <= d**2:
                            newly_infected[j] = True

        for i in range(n):
            if newly_infected[i]:
                infected[i] = True
                changed = True

    for i in range(n):
        print("Yes" if infected[i] else "No")

# YOUR CODE HERE
solve()