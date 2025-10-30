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
    
    # Now, for each bridge, compute the impact of closing it
    # The bridge between i and i+1 (or N and 1 for i=N)
    # We need to find for each bridge, the number of times it is used in the optimal path
    # Then, the new total distance is total - (number of times the bridge is used) + (number of times the alternative path is used)
    
    # To find the number of times each bridge is used, we need to find for each pair (X[i], X[i+1]), which bridges are used in the optimal path
    # For each pair, the optimal path is either the clockwise or counter-clockwise path
    # We can count the number of times each bridge is used in the optimal paths
    
    # Initialize a list to count the usage of each bridge
    bridge_usage = [0] * N
    
    for i in range(M-1):
        a = X[i]
        b = X[i+1]
        if a < b:
            dist1 = b - a
            dist2 = N - b + a
            if dist1 < dist2:
                # Use the clockwise path
                for j in range(a, b):
                    bridge_usage[j-1] += 1
            else:
                # Use the counter-clockwise path
                bridge_usage[N-1] += 1
                for j in range(b, a):
                    bridge_usage[j-1] += 1
        else:
            dist1 = a - b
            dist2 = N - a + b
            if dist1 < dist2:
                # Use the counter-clockwise path
                for j in range(b, a):
                    bridge_usage[j-1] += 1
            else:
                # Use the clockwise path
                bridge_usage[N-1] += 1
                for j in range(a, b):
                    bridge_usage[j-1] += 1
    
    # Now, for each bridge, compute the new total distance if it is closed
    # The new distance is total - bridge_usage[i] + (number of times the alternative path is used)
    # The alternative path is the other direction, so for each pair (X[j], X[j+1]) that used this bridge, we need to switch to the other path
    
    # To compute the alternative path, we need to find for each pair that used the bridge, the alternative distance
    # Then, the new total is total - bridge_usage[i] + sum(alternative distances for pairs that used the bridge)
    
    # To simplify, we can precompute for each pair, the alternative distance
    alternative_distances = []
    for i in range(M-1):
        a = X[i]
        b = X[i+1]
        if a < b:
            dist1 = b - a
            dist2 = N - b + a
            if dist1 < dist2:
                alternative_distances.append(dist2)
            else:
                alternative_distances.append(dist1)
        else:
            dist1 = a - b
            dist2 = N - a + b
            if dist1 < dist2:
                alternative_distances.append(dist2)
            else:
                alternative_distances.append(dist1)
    
    # Now, for each bridge, find the pairs that used it and compute the new total
    min_total = float('inf')
    for i in range(N):
        # Find the pairs that used bridge i
        # Bridge i connects i+1 and i+2 (mod N)
        # For each pair (X[j], X[j+1]), if the optimal path used bridge i, then we need to switch to the alternative path
        # To find which pairs used bridge i, we need to check for each pair if the optimal path includes bridge i
        # This is complex, so instead, we can precompute for each pair, the bridges used in the optimal path
        # Then, for each bridge, we can sum the alternative distances for pairs that used it
        
        # To do this efficiently, we can create a list of lists, where for each bridge, we store the indices of pairs that used it
        # However, this would require O(M*N) space, which is too much
        
        # Instead, we can precompute for each pair, the bridges used in the optimal path, and then for each bridge, sum the alternative distances for pairs that used it
        
        # Since we already have bridge_usage, which counts the number of times each bridge is used, we can use it to compute the new total
        
        # The new total is total - bridge_usage[i] + (sum of alternative distances for pairs that used bridge i)
        
        # To find the sum of alternative distances for pairs that used bridge i, we need to find for each pair, if it used bridge i, then add its alternative distance
        
        # Since we have the alternative_distances list, we can iterate over all pairs and check if they used bridge i
        
        # Initialize the sum of alternative distances
        sum_alt = 0
        for j in range(M-1):
            a = X[j]
            b = X[j+1]
            if a < b:
                dist1 = b - a
                dist2 = N - b + a
                if dist1 < dist2:
                    # Used clockwise path
                    if a <= i+1 < b:
                        sum_alt += alternative_distances[j]
                else:
                    # Used counter-clockwise path
                    if i+1 == N or (b <= i+1 < a):
                        sum_alt += alternative_distances[j]
            else:
                dist1 = a - b
                dist2 = N - a + b
                if dist1 < dist2:
                    # Used counter-clockwise path
                    if b <= i+1 < a:
                        sum_alt += alternative_distances[j]
                else:
                    # Used clockwise path
                    if i+1 == N or (a <= i+1 < b):
                        sum_alt += alternative_distances[j]
        
        new_total = total - bridge_usage[i] + sum_alt
        if new_total < min_total:
            min_total = new_total
    
    print(min_total)

if __name__ == "__main__":
    main()