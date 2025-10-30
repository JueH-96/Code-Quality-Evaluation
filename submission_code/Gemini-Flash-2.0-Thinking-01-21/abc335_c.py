def solve():
    n, q = map(int, input().split())
    x_coords = list(range(1, n + 1))
    y_coords = [0] * n
    
    for _ in range(q):
        query = input().split()
        query_type = int(query[0])
        if query_type == 1:
            direction = query[1]
            previous_x = x_coords[0]
            previous_y = y_coords[0]
            if direction == 'R':
                x_coords[0] += 1
            elif direction == 'L':
                x_coords[0] -= 1
            elif direction == 'U':
                y_coords[0] += 1
            elif direction == 'D':
                y_coords[0] -= 1
            for i in range(1, n):
                current_x = x_coords[i]
                current_y = y_coords[i]
                x_coords[i] = previous_x
                y_coords[i] = previous_y
                previous_x = current_x
                previous_y = current_y
        elif query_type == 2:
            part_index = int(query[1])
            print(x_coords[part_index-1], y_coords[part_index-1])

if __name__ == '__main__':
    solve()