def solve():
    n = int(input())
    h = list(map(int, input().split()))

    max_buildings = 1
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            diff = j - i
            height = h[i]
            
            next_index = j + diff
            while next_index < n:
                if h[next_index] == height:
                    count += 1
                next_index += diff
            max_buildings = max(max_buildings, count)
    print(max_buildings)

solve()