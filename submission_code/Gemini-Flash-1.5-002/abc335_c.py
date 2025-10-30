# YOUR CODE HERE
n, q = map(int, input().split())
parts = [(i, 0) for i in range(1, n + 1)]
for _ in range(q):
    query = list(map(str, input().split()))
    if query[0] == '1':
        direction = query[1]
        head_x, head_y = parts[0]
        if direction == 'R':
            head_x += 1
        elif direction == 'L':
            head_x -= 1
        elif direction == 'U':
            head_y += 1
        elif direction == 'D':
            head_y -= 1
        new_parts = [(head_x, head_y)]
        for i in range(1, n):
            new_parts.append(parts[i - 1])
        parts = new_parts
    else:
        p = int(query[1])
        print(parts[p - 1][0], parts[p - 1][1])