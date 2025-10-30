def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:]))
    
    # We need to find the minimum cost to sort the array
    # The cost of swapping P_i and P_{i+1} is i, so we want to minimize swaps with high costs
    
    # Create a position map to know where each number is located initially
    position = [0] * (N + 1)
    for i in range(N):
        position[P[i]] = i
    
    # We will use a greedy approach to bring each number to its correct position
    # from the smallest number to the largest number
    total_cost = 0
    i = 0
    while i < N:
        # If the number at position i is not i+1, we need to bring i+1 to this position
        while P[i] != i + 1:
            # The number that should be at position i is i+1
            # Find where i+1 is currently located
            correct_pos = position[i + 1]
            
            # Swap it with the number just before it until it reaches position i
            while correct_pos > i:
                # Swap P[correct_pos] with P[correct_pos - 1]
                # The cost of this swap is correct_pos - 1
                total_cost += correct_pos
                
                # Perform the swap in the array
                P[correct_pos], P[correct_pos - 1] = P[correct_pos - 1], P[correct_pos]
                
                # Update the positions in the position map
                position[P[correct_pos]], position[P[correct_pos - 1]] = position[P[correct_pos - 1]], position[P[correct_pos]]
                
                # Move the correct_pos pointer one step left
                correct_pos -= 1
        
        # Move to the next position
        i += 1
    
    print(total_cost)

if __name__ == "__main__":
    main()