def find_minimal_tour(N, M, stops):
    min_length = float('inf')
    
    # Try closing each bridge and find the minimum tour length
    for closed_bridge in range(1, N+1):
        length = 0
        
        for i in range(M-1):
            a, b = stops[i], stops[i+1]
            
            # Calculate positions on the line after removing the bridge
            if closed_bridge == N:  # Bridge between islands N and 1 is closed
                pos_a = a
                pos_b = b
            else:  # Bridge between islands closed_bridge and closed_bridge+1 is closed
                if a <= closed_bridge:
                    pos_a = N - closed_bridge + a
                else:
                    pos_a = a - closed_bridge
                
                if b <= closed_bridge:
                    pos_b = N - closed_bridge + b
                else:
                    pos_b = b - closed_bridge
            
            # Calculate distance (number of bridges crossed)
            length += abs(pos_a - pos_b)
        
        min_length = min(min_length, length)
    
    return min_length

# Read input
N, M = map(int, input().split())
stops = list(map(int, input().split()))

# Output the answer
print(find_minimal_tour(N, M, stops))