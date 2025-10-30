def main():
    import sys
    input_line = sys.stdin.readline().strip()
    if not input_line:
        return
    parts = input_line.split()
    N = int(parts[0])
    S = int(parts[1])
    M = int(parts[2])
    L = int(parts[3])
    
    # Maximum packs we might consider for each type:
    # We can go a bit over the minimum number required.
    max_x = (N // 6) + 5  # Pack of 6 eggs
    max_y = (N // 8) + 5  # Pack of 8 eggs
    max_z = (N // 12) + 5 # Pack of 12 eggs

    min_cost = float('inf')
    
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            for z in range(max_z + 1):
                eggs = 6 * x + 8 * y + 12 * z
                if eggs >= N:
                    cost = S * x + M * y + L * z
                    if cost < min_cost:
                        min_cost = cost
    sys.stdout.write(str(min_cost))

if __name__ == '__main__':
    main()