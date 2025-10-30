import sys

def main():
    n, q = map(int, sys.stdin.readline().split())
    head_positions = [(1, 0)]
    x, y = 1, 0
    m = 0  # number of moves done
    dir_map = {
        'R': (1, 0),
        'L': (-1, 0),
        'U': (0, 1),
        'D': (0, -1)
    }
    for _ in range(q):
        parts = sys.stdin.readline().split()
        if parts[0] == '1':
            c = parts[1]
            dx, dy = dir_map[c]
            x += dx
            y += dy
            head_positions.append((x, y))
            m += 1
        else:
            p = int(parts[1])
            if m >= p - 1:
                idx = m - (p - 1)
                print(head_positions[idx][0], head_positions[idx][1])
            else:
                print(p - m, 0)

if __name__ == "__main__":
    main()