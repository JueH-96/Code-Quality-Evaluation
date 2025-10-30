def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and D
    N, D = map(int, data[0].split())
    
    # Read thickness and length of each snake
    snakes = []
    for i in range(1, N + 1):
        T, L = map(int, data[i].split())
        snakes.append((T, L))
    
    # For each k from 1 to D, calculate the heaviest snake's weight
    results = []
    for k in range(1, D + 1):
        max_weight = 0
        for T, L in snakes:
            weight = T * (L + k)
            if weight > max_weight:
                max_weight = weight
        results.append(max_weight)
    
    # Print the results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()