def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    queries = data[2:]
    
    # Directions mapping
    dir_map = {'R': (1, 0), 'L': (-1, 0), 'U': (0, 1), 'D': (0, -1)}
    
    # Initialize prefix sum with (0,0)
    prefix_sum = [(0, 0)]
    
    # Initial head position
    initial_head_x = 1
    initial_head_y = 0
    
    t = 0  # movement step counter
    result = []
    
    for q in queries:
        if q == '1':
            # Movement query
            C = next(queries)
            dx, dy = dir_map[C]
            # New prefix sum
            px, py = prefix_sum[t]
            new_px = px + dx
            new_py = py + dy
            prefix_sum.append((new_px, new_py))
            t += 1
        elif q == '2':
            # Position query
            p = int(next(queries))
            if t - (p - 1) >= 0:
                px, py = prefix_sum[t - (p - 1)]
                x = initial_head_x + px + (p - 1) - 1
                y = initial_head_y + py + 0
            else:
                x = p
                y = 0
            result.append(f"{x} {y}")
    
    print('
'.join(result))

if __name__ == "__main__":
    main()