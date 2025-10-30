def solve():
    n = int(input())
    ranges = []
    for _ in range(n):
        ranges.append(list(map(int, input().split())))

    x = [0] * n
    total_sum = 0
    for i in range(n):
        x[i] = ranges[i][0]
        total_sum += x[i]

    diff = -total_sum
    
    
    indices = list(range(n))
    indices.sort(key=lambda i: ranges[i][1] - ranges[i][0], reverse=True)

    for i in indices:
        change = min(diff, ranges[i][1] - x[i])
        x[i] += change
        diff -= change
        if diff == 0:
            break

    if diff != 0:
        print("No")
    else:
        print("Yes")
        print(*x)

solve()