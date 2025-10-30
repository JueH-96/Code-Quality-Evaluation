def solve():
    n = int(input())
    h = list(map(int, input().split()))

    max_buildings = 1

    for i in range(n):
        # Consider single building case
        max_buildings = max(max_buildings, 1)
        for j in range(i + 1, n):
            if h[i] == h[j]:
                interval = j - i
                count = 2
                next_index = j + interval
                while next_index < n:
                    if h[next_index] == h[i]:
                        count += 1
                    next_index += interval
                max_buildings = max(max_buildings, count)

    print(max_buildings)

solve()