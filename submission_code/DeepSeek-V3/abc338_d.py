# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    X = list(map(int, data[2:2+M]))
    
    # Precompute the distances between consecutive Xs
    distances = []
    for i in range(M-1):
        a = X[i]
        b = X[i+1]
        if a < b:
            dist1 = b - a
            dist2 = N - b + a
        else:
            dist1 = a - b
            dist2 = N - a + b
        distances.append(min(dist1, dist2))
    
    # Precompute the sum of all distances
    total = sum(distances)
    
    # Now, for each bridge, calculate the impact of closing it
    # The bridge between i and i+1 (or N and 1 for i=N)
    # We need to find for each bridge, the number of times it is used in the optimal path
    # The optimal path between two consecutive Xs is the minimum of the two possible paths
    # So, for each bridge, we count how many times it is used in the optimal paths
    
    # To do this, we need to find for each bridge, the number of pairs (X[j], X[j+1]) where the bridge is on the optimal path
    # For bridge between i and i+1 (or N and 1 for i=N), it is used if:
    # For a pair (a, b), the bridge is on the path a -> b or b -> a, depending on the direction
    
    # To count this, we can precompute for each bridge, the number of pairs where it is on the optimal path
    # For bridge between i and i+1, it is used if:
    # a <= i < b or b <= i < a, and the optimal path is the one that uses this bridge
    
    # To implement this, we can precompute for each bridge, the number of pairs where it is on the optimal path
    # We can use a list to count the usage of each bridge
    bridge_usage = [0] * N
    
    for j in range(M-1):
        a = X[j]
        b = X[j+1]
        if a < b:
            # The optimal path is either a -> b (clockwise) or a -> 1 -> N -> b (counter-clockwise)
            # The bridge between i and i+1 is used in the clockwise path if a <= i < b
            # So, we increment the usage for i in [a, b-1]
            for i in range(a, b):
                bridge_usage[i % N] += 1
        else:
            # The optimal path is either a -> b (counter-clockwise) or a -> N -> 1 -> b (clockwise)
            # The bridge between i and i+1 is used in the counter-clockwise path if b <= i < a
            # So, we increment the usage for i in [b, a-1]
            for i in range(b, a):
                bridge_usage[i % N] += 1
    
    # Now, for each bridge, the impact of closing it is the number of times it is used in the optimal paths
    # The new total distance is total - bridge_usage[i] + (number of pairs where the bridge is not used in the optimal path)
    # Wait, no. When a bridge is closed, the path between two consecutive Xs that used that bridge must now take the other path
    # So, for each pair that used the bridge, the distance increases by the difference between the two paths
    
    # So, for each bridge, the new total distance is:
    # total - bridge_usage[i] * (min_dist) + bridge_usage[i] * (max_dist)
    # where min_dist is the original distance, and max_dist is the alternative distance
    
    # To compute this, we need to know for each pair, the original distance and the alternative distance
    # So, we need to precompute for each pair, the two possible distances
    
    # Let's precompute for each pair, the two distances
    pair_distances = []
    for j in range(M-1):
        a = X[j]
        b = X[j+1]
        if a < b:
            dist1 = b - a
            dist2 = N - b + a
        else:
            dist1 = a - b
            dist2 = N - a + b
        pair_distances.append((min(dist1, dist2), max(dist1, dist2)))
    
    # Now, for each bridge, the impact is:
    # For each pair that used the bridge, the distance increases by (max_dist - min_dist)
    # So, the new total distance is total + bridge_usage[i] * (max_dist - min_dist)
    
    # To compute this, we need to know for each pair, whether it used the bridge
    # So, for each pair, we need to know which bridges are on its optimal path
    
    # To do this, we can precompute for each pair, the set of bridges on its optimal path
    # Then, for each bridge, we can count how many pairs have it in their optimal path
    
    # Alternatively, we can precompute for each bridge, the list of pairs that use it
    
    # Given the constraints, we need an efficient way to compute this
    
    # Let's precompute for each bridge, the list of pairs that use it
    bridge_pairs = [[] for _ in range(N)]
    for j in range(M-1):
        a = X[j]
        b = X[j+1]
        if a < b:
            for i in range(a, b):
                bridge_pairs[i % N].append(j)
        else:
            for i in range(b, a):
                bridge_pairs[i % N].append(j)
    
    # Now, for each bridge, we can compute the impact
    min_total = float('inf')
    for i in range(N):
        impact = 0
        for j in bridge_pairs[i]:
            min_dist, max_dist = pair_distances[j]
            impact += (max_dist - min_dist)
        new_total = total + impact
        if new_total < min_total:
            min_total = new_total
    
    print(min_total)

if __name__ == "__main__":
    main()