def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = int(data[1])
    M = int(data[2])
    L = int(data[3])
    
    max_eggs = N + 11
    max_a = (max_eggs) // 6 + 1
    max_b = (max_eggs) // 8 + 1
    max_c = (max_eggs) // 12 + 1
    
    min_cost = float('inf')
    
    for a in range(0, max_a + 1):
        for b in range(0, max_b + 1):
            base_eggs = 6 * a + 8 * b
            if base_eggs > max_eggs:
                continue
            for c in range(0, max_c + 1):
                total_eggs = base_eggs + 12 * c
                if total_eggs > max_eggs:
                    break
                if total_eggs >= N:
                    cost = a * S + b * M + c * L
                    if cost < min_cost:
                        min_cost = cost
    print(min_cost)

if __name__ == '__main__':
    main()