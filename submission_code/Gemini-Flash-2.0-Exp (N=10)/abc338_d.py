def solve():
    n, m = map(int, input().split())
    x = list(map(int, input().split()))

    def calculate_distance(start, end, closed_bridge):
        dist1 = abs(end - start)
        dist2 = n - dist1
        
        if start > end:
            start, end = end, start
        
        if closed_bridge == start and closed_bridge != n:
            return dist2
        elif closed_bridge == end and closed_bridge != n:
            return dist2
        elif closed_bridge == n and (start == 1 and end == n or start == n and end == 1):
            return dist1
        else:
            return min(dist1, dist2)

    min_tour_length = float('inf')
    for closed_bridge in range(1, n + 1):
        tour_length = 0
        for i in range(m - 1):
            tour_length += calculate_distance(x[i], x[i+1], closed_bridge)
        min_tour_length = min(min_tour_length, tour_length)
    print(min_tour_length)

solve()