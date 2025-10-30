def solve():
    n, t = map(int, input().split())
    s = input()
    x = list(map(int, input().split()))

    ants = []
    for i in range(n):
        ants.append((x[i], int(s[i])))

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            pos1, dir1 = ants[i]
            pos2, dir2 = ants[j]

            if dir1 == dir2:
                continue
            
            if dir1 == 0 and dir2 == 1:
                if pos1 > pos2:
                    time = (pos1 - pos2) / 2
                    if time <= t:
                        count += 1
            elif dir1 == 1 and dir2 == 0:
                if pos1 < pos2:
                    time = (pos2 - pos1) / 2
                    if time <= t:
                        count += 1
    print(count)

solve()