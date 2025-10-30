def solve():
    n, q = map(int, input().split())
    x_coords = [i+1 for i in range(n)]
    y_coords = [0] * n
    
    for _ in range(q):
        query = input().split()
        query_type = int(query[0])
        if query_type == 1:
            direction = query[1]
            prev_x = x_coords[0]
            prev_y = y_coords[0]
            if direction == 'R':
                x_coords[0] += 1
            elif direction == 'L':
                x_coords[0] -= 1
            elif direction == 'U':
                y_coords[0] += 1
            elif direction == 'D':
                y_coords[0] -= 1
            for i in range(1, n):
                temp_x = x_coords[i]
                temp_y = y_coords[i]
                x_coords[i] = prev_x
                y_coords[i] = prev_y
                prev_x = temp_x
                prev_y = temp_y
        elif query_type == 2:
            p = int(query[1])
            print(x_coords[p-1], y_coords[p-1])

if __name__ == '__main__':
    solve()