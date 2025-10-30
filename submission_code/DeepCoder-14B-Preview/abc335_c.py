from collections import deque
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    # Initialize the deque with positions (1,0), (2,0), ..., (N,0)
    positions = deque()
    for i in range(1, N+1):
        positions.append((i, 0))

    output = []
    for _ in range(Q):
        query = input[ptr]
        ptr += 1
        if query == '1':
            # It's a move command
            C = input[ptr]
            ptr += 1
            dx, dy = 0, 0
            if C == 'R':
                dx = 1
            elif C == 'L':
                dx = -1
            elif C == 'U':
                dy = 1
            elif C == 'D':
                dy = -1
            else:
                pass  # should not happen

            current_head = positions[0]
            new_head = (current_head[0] + dx, current_head[1] + dy)
            positions.appendleft(new_head)
            positions.pop()
        else:
            # It's a query command
            p = int(input[ptr])
            ptr += 1
            x, y = positions[p-1]
            output.append(f"{x} {y}")

    print('
'.join(output))

if __name__ == '__main__':
    main()