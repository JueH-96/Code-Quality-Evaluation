def solve():
    n = int(input())
    heights = list(map(int, input().split()))
    counts = []
    for i in range(n):
        count = 0
        for j in range(i + 1, n):
            max_height_between = -1
            is_taller_building = False
            for k in range(i + 1, j):
                if heights[k] > heights[j]:
                    is_taller_building = True
                    break
            if not is_taller_building:
                count += 1
        counts.append(count)
    print(*(counts))

solve()