def solve():
    n, d = map(int, input().split())
    points = []
    for _ in range(n):
        points.append(list(map(int, input().split())))

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    def calculate_f(x, x_coords):
        return sum(abs(x - xi) for xi in x_coords)

    def calculate_g(y, y_coords):
        return sum(abs(y - yi) for yi in y_coords)

    count = 0
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    search_min_x = min_x - d - 1
    search_max_x = max_x + d + 1
    search_min_y = min_y - d - 1
    search_max_y = max_y + d + 1

    for x in range(search_min_x, search_max_x + 1):
        fx = calculate_f(x, x_coords)
        if fx <= d:
            target_g = d - fx

            def check_g(y):
                return calculate_g(y, y_coords) <= target_g

            low = search_min_y
            high = search_max_y
            y_min = None
            while low <= high:
                mid = (low + high) // 2
                if check_g(mid):
                    y_min = mid
                    high = mid - 1
                else:
                    low = mid + 1

            low = search_min_y
            high = search_max_y
            y_max = None
            while low <= high:
                mid = (low + high) // 2
                if check_g(mid):
                    y_max = mid
                    low = mid + 1
                else:
                    high = mid - 1

            if y_min is not None and y_max is not None:
                count += (y_max - y_min + 1)

    print(count)

solve()