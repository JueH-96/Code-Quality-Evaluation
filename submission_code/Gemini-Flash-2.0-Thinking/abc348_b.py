def solve():
    n = int(input())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append((x, y))

    for i in range(n):
        max_squared_distance = -1
        farthest_point_id = -1
        x1, y1 = points[i]
        for j in range(n):
            if i == j:
                continue
            x2, y2 = points[j]
            squared_distance = (x1 - x2)**2 + (y1 - y2)**2
            current_point_id = j + 1

            if squared_distance > max_squared_distance:
                max_squared_distance = squared_distance
                farthest_point_id = current_point_id
            elif squared_distance == max_squared_distance and current_point_id < farthest_point_id:
                farthest_point_id = current_point_id
            elif squared_distance == max_squared_distance and farthest_point_id == -1:
                farthest_point_id = current_point_id

        print(farthest_point_id)

solve()