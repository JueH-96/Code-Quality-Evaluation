def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    # Read input values: N eggs required, cost S for 6 eggs, M for 8 eggs, L for 12 eggs.
    N = int(data[0])
    S = int(data[1])
    M = int(data[2])
    L = int(data[3])

    # Set an initially high cost.
    min_cost = float('inf')
    
    # Establish search bounds for each pack type.
    # We add a few extra iterations to cover the case when extra packs are needed to reach at least N eggs.
    max_a = N // 6 + 3
    max_b = N // 8 + 3
    max_c = N // 12 + 3

    # Brute force through all combinations.
    for a in range(max_a):
        for b in range(max_b):
            for c in range(max_c):
                eggs = 6 * a + 8 * b + 12 * c
                if eggs >= N:
                    cost = S * a + M * b + L * c
                    if cost < min_cost:
                        min_cost = cost
                        
    print(min_cost)

if __name__ == '__main__':
    main()