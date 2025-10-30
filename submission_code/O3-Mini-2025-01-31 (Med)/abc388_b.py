def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    # Parse input values
    N = int(data[0])
    D = int(data[1])
    snakes = []
    
    index = 2
    for _ in range(N):
        T = int(data[index])
        L = int(data[index + 1])
        snakes.append((T, L))
        index += 2

    # For each k from 1 to D, compute the maximum weight among all snakes
    results = []
    for k in range(1, D + 1):
        max_weight = 0
        for T, L in snakes:
            # Each snake's weight when its length is increased by k
            weight = T * (L + k)
            if weight > max_weight:
                max_weight = weight
        results.append(str(max_weight))
    
    # Output the results, one per line
    sys.stdout.write("
".join(results))

if __name__ == "__main__":
    main()