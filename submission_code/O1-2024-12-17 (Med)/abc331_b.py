def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, S, M, L = map(int, data)

    # We'll try all combinations of packs such that 6*x + 8*y + 12*z >= N
    # N <= 100, so we won't exceed reasonable bounds.
    
    min_cost = float('inf')
    # The upper bounds for x, y, z are chosen so that we definitely cover N eggs
    max_x = (N // 6) + 2
    max_y = (N // 8) + 2
    max_z = (N // 12) + 2
    
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            for z in range(max_z + 1):
                eggs = 6*x + 8*y + 12*z
                if eggs >= N:
                    cost = S*x + M*y + L*z
                    if cost < min_cost:
                        min_cost = cost

    print(min_cost)

if __name__ == "__main__":
    main()