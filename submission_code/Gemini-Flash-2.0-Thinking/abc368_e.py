def solve():
    n, m, x1_val = map(int, input().split())
    trains = []
    for _ in range(m):
        trains.append(list(map(int, input().split())))

    x = [0] * m
    x[0] = x1_val

    changed = True
    while changed:
        changed = False
        for i in range(m):
            for j in range(m):
                if trains[i][1] == trains[j][0] and trains[i][3] <= trains[j][2]:
                    required_xj = trains[i][3] + x[i] - trains[j][2]
                    if required_xj > x[j]:
                        x[j] = required_xj
                        changed = True

    print(*x[1:])

if __name__ == "__main__":
    solve()