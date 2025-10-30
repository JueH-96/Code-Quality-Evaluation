def main():
    import sys

    # Read input
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N, S, M, L = map(int, data)

    # Define cost list using (eggs, cost)
    packs = [(6, S), (8, M), (12, L)]
    
    # Since N is at most 100, we can brute force over possible numbers of packs.
    # We need to set limits for the loops. We set generous bounds based on egg counts.
    # Maximum number of any pack is enough if we take all eggs from that pack.
    max6 = (N // 6) + 2
    max8 = (N // 8) + 2
    max12 = (N // 12) + 2
    
    min_cost = float('inf')
    
    for a in range(max6):
        for b in range(max8):
            for c in range(max12):
                eggs = a * 6 + b * 8 + c * 12
                if eggs >= N:
                    cost = a * S + b * M + c * L
                    if cost < min_cost:
                        min_cost = cost

    # Print the answer
    print(min_cost)

if __name__ == '__main__':
    main()