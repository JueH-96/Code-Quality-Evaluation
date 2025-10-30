def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = list(map(int, data[1:N+1]))
    L1 = int(data[N+1])
    C1 = int(data[N+2])
    K1 = int(data[N+3])
    L2 = int(data[N+4])
    C2 = int(data[N+5])
    K2 = int(data[N+6])
    
    # Initialize the minimum cost to a large number
    min_cost = float('inf')
    
    # Iterate over possible counts of type-1 sensors
    for x1 in range(K1 + 1):
        # Iterate over possible counts of type-2 sensors
        for x2 in range(K2 + 1):
            # Check if all sections can be monitored with x1 and x2 sensors
            possible = True
            for d in D:
                # Calculate the maximum length that can be monitored with x1 and x2 sensors
                max_length = x1 * L1 + x2 * L2
                if max_length < d:
                    possible = False
                    break
            if possible:
                # Calculate the total cost
                total_cost = x1 * C1 + x2 * C2
                if total_cost < min_cost:
                    min_cost = total_cost
    
    if min_cost != float('inf'):
        print(min_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()