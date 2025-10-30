import sys

n, q = map(int, sys.stdin.readline().split())

head_positions = [(1, 0)]  # After 0 moves

for _ in range(q):
    parts = sys.stdin.readline().split()
    if parts[0] == '1':
        # Move command
        direction = parts[1]
        dx, dy = 0, 0
        if direction == 'R':
            dx = 1
        elif direction == 'L':
            dx = -1
        elif direction == 'U':
            dy = 1
        elif direction == 'D':
            dy = -1
        new_head = (head_positions[-1][0] + dx, head_positions[-1][1] + dy)
        head_positions.append(new_head)
    else:
        # Query command
        p = int(parts[1])
        m = len(head_positions) - 1  # Current number of moves
        if m >= p - 1:
            k = m - (p - 1)
            x, y = head_positions[k]
        else:
            x = p - m
            y = 0
        print(x, y)