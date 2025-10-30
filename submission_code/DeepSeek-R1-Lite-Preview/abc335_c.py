def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    queries = data[2:]
    
    head_positions = [(1, 0)]  # head position after 0 moves
    m = 0  # current move count
    
    dir_map = {
        'R': (1, 0),
        'L': (-1, 0),
        'U': (0, 1),
        'D': (0, -1)
    }
    
    output = []
    
    idx = 0
    for _ in range(Q):
        if queries[idx] == '1':
            C = queries[idx + 1]
            V = dir_map[C]
            new_x = head_positions[m][0] + V[0]
            new_y = head_positions[m][1] + V[1]
            head_positions.append((new_x, new_y))
            m += 1
            idx += 2
        else:
            p = int(queries[idx + 1])
            k = m - p + 1
            if k >= 0:
                pos = head_positions[k]
            else:
                pos = (p, 0)
            output.append(f"{pos[0]} {pos[1]}")
            idx += 2
    
    print("
".join(output))

if __name__ == "__main__":
    main()