# YOUR CODE HERE
def solve():
    n, d = map(int, input().split())
    coords = []
    for _ in range(n):
        coords.append(list(map(int, input().split())))

    infected = [False] * n
    infected[0] = True

    q = [0]
    while q:
        curr = q.pop(0)
        for i in range(n):
            if not infected[i]:
                dist_sq = (coords[curr][0] - coords[i][0])**2 + (coords[curr][1] - coords[i][1])**2
                if dist_sq <= d**2:
                    infected[i] = True
                    q.append(i)

    for i in range(n):
        if infected[i]:
            print("Yes")
        else:
            print("No")

solve()