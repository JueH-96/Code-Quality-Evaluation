def minimum_tour_length(N, M, X):
    # Create a list to store the positions of the bridges
    positions = []
    
    # Calculate the positions of the bridges in the tour
    for i in range(M):
        positions.append(X[i] - 1)  # Convert to 0-based index
    
    # Initialize the minimum length to a large number
    min_length = float('inf')
    
    # Check the length of the tour for each bridge closure
    for bridge in range(N):
        length = 0
        last_position = -1
        
        for pos in positions:
            if last_position == -1:
                length += 1  # First position
            else:
                if (last_position + 1) % N == pos:
                    length += 1  # Directly connected
                elif (last_position - 1 + N) % N == pos:
                    length += 1  # Directly connected
                else:
                    # Need to go around the closed bridge
                    if (last_position + 1) % N == bridge:
                        length += 2  # Go around the closed bridge
                    elif (last_position - 1 + N) % N == bridge:
                        length += 2  # Go around the closed bridge
                    else:
                        length += 1  # Directly connected
            
            last_position = pos
        
        min_length = min(min_length, length)
    
    return min_length

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
X = list(map(int, data[2:M+2]))

result = minimum_tour_length(N, M, X)
print(result)