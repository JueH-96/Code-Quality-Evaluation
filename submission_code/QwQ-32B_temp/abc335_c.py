import sys

def main():
    N, Q = map(int, sys.stdin.readline().split())
    head_positions = [(1, 0)]  # Initial position of the head

    for _ in range(Q):
        parts = sys.stdin.readline().split()
        if parts[0] == '1':
            # Move command
            direction = parts[1]
            x, y = head_positions[-1]
            if direction == 'R':
                new_x, new_y = x + 1, y
            elif direction == 'L':
                new_x, new_y = x - 1, y
            elif direction == 'U':
                new_x, new_y = x, y + 1
            else:  # D
                new_x, new_y = x, y - 1
            head_positions.append((new_x, new_y))
        else:
            # Query for position of part p
            p = int(parts[1])
            current_move = len(head_positions) - 1
            step_needed = current_move - (p - 1)
            if step_needed >= 0:
                x, y = head_positions[step_needed]
            else:
                x = p - current_move
                y = 0
            print(x, y)

if __name__ == "__main__":
    main()