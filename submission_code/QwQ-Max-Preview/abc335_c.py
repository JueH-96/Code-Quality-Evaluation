import sys

def main():
    n, q = map(int, sys.stdin.readline().split())
    history = []
    t = 0
    for _ in range(q):
        parts = sys.stdin.readline().split()
        if parts[0] == '1':
            # Move the head
            c = parts[1]
            if not history:
                current_x, current_y = 1, 0
            else:
                current_x, current_y = history[-1]
            if c == 'R':
                new_x = current_x + 1
                new_y = current_y
            elif c == 'L':
                new_x = current_x - 1
                new_y = current_y
            elif c == 'U':
                new_x = current_x
                new_y = current_y + 1
            elif c == 'D':
                new_x = current_x
                new_y = current_y - 1
            history.append((new_x, new_y))
            t += 1
        else:
            # Query part p
            p = int(parts[1])
            if t >= p - 1:
                m = t - (p - 1)
                if m == 0:
                    x, y = 1, 0
                else:
                    x, y = history[m - 1]
            else:
                x = p - t
                y = 0
            print(f"{x} {y}")

if __name__ == "__main__":
    main()