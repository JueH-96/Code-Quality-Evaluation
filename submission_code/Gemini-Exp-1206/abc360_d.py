def solve():
    n, t = map(int, input().split())
    s = input()
    x = list(map(int, input().split()))
    
    ants = []
    for i in range(n):
        ants.append((x[i], int(s[i]), i))
    
    ants.sort()
    
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ants[i][1] == 1 and ants[j][1] == 0:
                if (ants[j][0] - ants[i][0]) / 2 <= t:
                    count += 1
    
    print(count)

solve()