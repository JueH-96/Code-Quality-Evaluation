def minimum_cost_to_sort(N, P):
    # Create a list to store the position of each number in the permutation
    position = [0] * (N + 1)
    
    # Fill the position array with the indices of each number in P
    for i in range(N):
        position[P[i]] = i
    
    total_cost = 0
    
    # We will iterate from the largest number to the smallest
    for number in range(N, 0, -1):
        # Get the current position of the number
        current_pos = position[number]
        
        # We need to move this number to its sorted position (number - 1)
        # The sorted position for number is (number - 1)
        while current_pos > number - 1:
            # Swap with the previous element
            total_cost += current_pos  # Cost of the swap
            # Update the positions in the position array
            position[number], position[P[current_pos - 1]] = position[P[current_pos - 1]], position[number]
            # Swap in the permutation
            P[current_pos], P[current_pos - 1] = P[current_pos - 1], P[current_pos]
            # Move to the previous position
            current_pos -= 1
    
    return total_cost

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = list(map(int, data[1:]))

result = minimum_cost_to_sort(N, P)
print(result)