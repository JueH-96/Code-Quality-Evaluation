import sys

def main():
    n, q = map(int, sys.stdin.readline().split())
    history = [(1, 0)]  # initial position of the head
    for _ in range(q):
        parts = sys.stdin.readline().split()
        if parts[0] == '1':
            direction = parts[1]
            x, y = history[-1]
            if direction == 'R':
                x += 1
            elif direction == 'L':
                x -= 1
            elif direction == 'U':
                y += 1
            elif direction == 'D':
                y -= 1
            history.append((x, y))
        else:
            p = int(parts[1])
            m = len(history) - 1
            if m >= p - 1:
                idx = m - (p - 1)
                x_out, y_out = history[idx]
            else:
                x_out = 1 + (p - 1 - m)
                y_out = 0
            print(x_out, y_out)

if __name__ == "__main__":
    main()