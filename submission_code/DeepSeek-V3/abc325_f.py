# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = list(map(int, data[1:N+1]))
    L1, C1, K1 = map(int, data[N+1:N+4])
    L2, C2, K2 = map(int, data[N+4:N+7])
    
    # Initialize the minimum cost to a large number
    min_cost = float('inf')
    
    # Iterate over possible counts of type-1 sensors
    for a in range(K1 + 1):
        # Iterate over possible counts of type-2 sensors
        for b in range(K2 + 1):
            # Check if all sections can be covered
            possible = True
            for d in D:
                # Calculate the maximum length that can be covered with a type-1 and b type-2 sensors
                max_length = a * L1 + b * L2
                if max_length < d:
                    possible = False
                    break
            if possible:
                # Calculate the total cost
                total_cost = a * C1 + b * C2
                if total_cost < min_cost:
                    min_cost = total_cost
    
    if min_cost != float('inf'):
        print(min_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()