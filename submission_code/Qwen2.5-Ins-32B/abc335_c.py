import sys

N, Q = map(int, input().split())
positions = [(i, 0) for i in range(1, N + 1)]

for _ in range(Q):
    query = input().split()
    if query[0] == '1':
        direction = query[1]
        prev_pos = positions[0]
        if direction == 'R':
            positions[0] = (positions[0][0] + 1, positions[0][1])
        elif direction == 'L':
            positions[0] = (positions[0][0] - 1, positions[0][1])
        elif direction == 'U':
            positions[0] = (positions[0][0], positions[0][1] + 1)
        elif direction == 'D':
            positions[0] = (positions[0][0], positions[0][1] - 1)
        
        for i in range(1, N):
            current_pos = positions[i]
            if current_pos == prev_pos:
                break
            positions[i] = prev_pos
            prev_pos = current_pos
    else:
        part = int(query[1])
        print(positions[part - 1][0], positions[part - 1][1])