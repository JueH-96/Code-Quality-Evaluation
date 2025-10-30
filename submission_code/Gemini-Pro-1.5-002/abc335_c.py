# YOUR CODE HERE
n, q = map(int, input().split())
pos = [(i, 0) for i in range(1, n + 1)]

for _ in range(q):
    query = input().split()
    if query[0] == '1':
        c = query[1]
        dx, dy = 0, 0
        if c == 'R':
            dx = 1
        elif c == 'L':
            dx = -1
        elif c == 'U':
            dy = 1
        elif c == 'D':
            dy = -1
        
        new_pos = [(pos[0][0] + dx, pos[0][1] + dy)]
        for i in range(1, n):
            new_pos.append(pos[i-1])
        pos = new_pos
    else:
        p = int(query[1])
        print(pos[p-1][0], pos[p-1][1])