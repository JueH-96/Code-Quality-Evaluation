def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    X = list(map(int, data[2:]))
    
    # Calculate the initial tour length without closing any bridge
    initial_tour_length = 0
    for i in range(M - 1):
        initial_tour_length += min(abs(X[i] - X[i + 1]), N - abs(X[i] - X[i + 1]))
    
    # Calculate the impact of closing each bridge
    bridge_impact = [0] * N
    
    for i in range(M - 1):
        a = X[i]
        b = X[i + 1]
        if a > b:
            a, b = b, a
        # Direct distance
        direct_distance = b - a
        # Alternative distance if the direct bridge is closed
        alternative_distance = N - direct_distance
        # Impact of closing the bridge between a and b
        bridge_impact[a] += alternative_distance - direct_distance
        bridge_impact[b] -= alternative_distance - direct_distance
    
    # Calculate the minimum possible tour length
    min_tour_length = initial_tour_length
    current_impact = 0
    
    for i in range(N):
        current_impact += bridge_impact[i]
        min_tour_length = min(min_tour_length, initial_tour_length + current_impact)
    
    print(min_tour_length)