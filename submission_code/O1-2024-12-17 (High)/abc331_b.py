def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, S, M, L = map(int, data)
    
    min_cost = float('inf')
    
    # We only need to consider buying up to around 20 packs of each type
    # since that easily covers eggs up to at least 100.
    for a in range(21):
        for b in range(21):
            for c in range(21):
                eggs = 6*a + 8*b + 12*c
                if eggs >= N:
                    cost = a*S + b*M + c*L
                    if cost < min_cost:
                        min_cost = cost
    
    print(min_cost)

main()