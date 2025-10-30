import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n, q = map(int, data[0].split())
    
    buf_x = [0] * n
    buf_y = [0] * n
    
    T = 0
    head_x, head_y = 1, 0
    buf_x[0] = 1
    buf_y[0] = 0
    
    output_lines = []
    index = 1
    for _ in range(q):
        parts = data[index].split()
        index += 1
        if parts[0] == '1':
            c = parts[1]
            T += 1
            if c == 'R':
                head_x += 1
            elif c == 'L':
                head_x -= 1
            elif c == 'U':
                head_y += 1
            elif c == 'D':
                head_y -= 1
            store_index = T % n
            buf_x[store_index] = head_x
            buf_y[store_index] = head_y
        else:
            p = int(parts[1])
            if T >= p - 1:
                k = T - (p - 1)
                ret_index = k % n
                x = buf_x[ret_index]
                y = buf_y[ret_index]
                output_lines.append(f"{x} {y}")
            else:
                x = p - T
                y = 0
                output_lines.append(f"{x} {y}")
                
    print("
".join(output_lines))

if __name__ == "__main__":
    main()