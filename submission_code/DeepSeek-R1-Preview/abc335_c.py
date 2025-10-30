def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    
    positions = [(1, 0)]  # positions[0] is initial position, positions[1] after first move, etc.
    M = 0  # number of moves
    
    output = []
    
    for _ in range(Q):
        query_type = data[idx]
        idx += 1
        if query_type == '1':
            # Move the head
            C = data[idx]
            idx += 1
            current_x, current_y = positions[-1]
            if C == 'R':
                current_x += 1
            elif C == 'L':
                current_x -= 1
            elif C == 'U':
                current_y += 1
            elif C == 'D':
                current_y -= 1
            positions.append((current_x, current_y))
            M += 1
        else:
            # Query
            p = int(data[idx])
            idx += 1
            if (p - 1) > M:
                x = p - M
                y = 0
            else:
                k = M - (p - 1)
                x, y = positions[k]
            output.append(f"{x} {y}")
    
    print('
'.join(output))

if __name__ == '__main__':
    main()