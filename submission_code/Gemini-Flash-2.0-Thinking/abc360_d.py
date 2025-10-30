def solve():
    n, t = map(int, input().split())
    s = input()
    x = list(map(int, input().split()))

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi = x[i]
            xj = x[j]
            si = s[i]
            sj = s[j]

            if xi < xj:
                if si == '1' and sj == '0':
                    time_to_pass = (xj - xi) / 2.0
                    if 0 < time_to_pass < t + 0.1:
                        count += 1
            elif xi > xj:
                if si == '0' and sj == '1':
                    time_to_pass = (xi - xj) / 2.0
                    if 0 < time_to_pass < t + 0.1:
                        count += 1
    print(count)

solve()