def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    X = list(map(int, data[2:2+M]))
    
    # Calculate the total distance without any bridge removal
    total_distance = 0
    for i in range(M - 1):
        if X[i] < X[i + 1]:
            total_distance += X[i + 1] - X[i]
        else:
            total_distance += X[i + 1] - X[i] + N
    
    # Calculate the impact of removing each bridge
    # We need to consider the effect of removing the bridge between i and i+1 for each i in 1 to N
    # where we consider N+1 as 1 (circular).
    # We use a list to accumulate the effects of not using each bridge.
    bridge_effect = [0] * N
    
    for i in range(M - 1):
        current = X[i]
        next_island = X[i + 1]
        
        if current < next_island:
            # Normal forward movement
            bridge_effect[current] += 1
            bridge_effect[next_island] -= 1
        else:
            # Wrapping around the circular structure
            bridge_effect[current] += 1
            bridge_effect[1] += 1
            bridge_effect[next_island] -= 1
    
    # Now, accumulate the effects to see the net effect of not using each bridge
    current_effect = 0
    min_additional_cost = float('inf')
    
    for i in range(1, N+1):
        current_effect += bridge_effect[i]
        # Since we're looking for the minimum tour length after removing one bridge,
        # we want to minimize the additional cost induced by not using a bridge.
        # `current_effect` tells us how many more steps we need if we don't use bridge i.
        min_additional_cost = min(min_additional_cost, current_effect)
    
    # The minimum possible length of the tour when the bridge to be closed is chosen optimally
    min_possible_length = total_distance - min_additional_cost
    
    print(min_possible_length)

if __name__ == "__main__":
    main()