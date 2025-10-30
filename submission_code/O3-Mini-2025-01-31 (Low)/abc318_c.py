def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    D = int(next(it))
    P = int(next(it))
    fares = [int(next(it)) for _ in range(N)]
    
    # Sort the fares in descending order.
    fares.sort(reverse=True)
    
    total_cost = 0
    # Process days in groups of D (the size of one batch).
    # For each group, the cost if paying regular fares is the sum of that group's fares.
    # But if we buy a batch of D one-day passes, it costs P.
    # So, for that group, we take the cheaper option.
    for i in range(0, N, D):
        group_sum = sum(fares[i:i+D])
        total_cost += min(P, group_sum)
    
    sys.stdout.write(str(total_cost))
    
if __name__ == '__main__':
    main()