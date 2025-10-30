# YOUR CODE HERE
import sys
import io

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    Q = int(input[idx + 1])
    idx += 2

    positions = [(i, 0) for i in range(1, N + 1)]
    directions = {'R': (1, 0), 'L': (-1, 0), 'U': (0, 1), 'D': (0, -1)}
    results = []

    for _ in range(Q):
        query_type = input[idx]
        if query_type == '1':
            C = input[idx + 1]
            dx, dy = directions[C]
            new_head_x = positions[0][0] + dx
            new_head_y = positions[0][1] + dy
            positions[0] = (new_head_x, new_head_y)
            for i in range(1, N):
                positions[i], prev_pos = prev_pos, positions[i]
            idx += 2
        elif query_type == '2':
            p = int(input[idx + 1])
            results.append(positions[p - 1])
            idx += 2

    for x, y in results:
        print(x, y)

if __name__ == "__main__":
    main()