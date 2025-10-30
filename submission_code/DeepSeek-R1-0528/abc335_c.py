import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n, q = map(int, data[0].split())
    t = 0
    head_history = [(1, 0)]
    output_lines = []
    
    for i in range(1, q + 1):
        parts = data[i].split()
        if parts[0] == '1':
            t += 1
            dir_char = parts[1]
            last_x, last_y = head_history[-1]
            if dir_char == 'R':
                new_head = (last_x + 1, last_y)
            elif dir_char == 'L':
                new_head = (last_x - 1, last_y)
            elif dir_char == 'U':
                new_head = (last_x, last_y + 1)
            elif dir_char == 'D':
                new_head = (last_x, last_y - 1)
            head_history.append(new_head)
        else:
            p = int(parts[1])
            idx = t - (p - 1)
            if idx < 0:
                x = 1 - idx
                y = 0
            else:
                x, y = head_history[idx]
            output_lines.append(f"{x} {y}")
    
    print("
".join(output_lines))

if __name__ == "__main__":
    main()