def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    T = int(data[0])
    index = 1
    results = []
    
    for _ in range(T):
        K = int(data[index])
        S_x = int(data[index + 1])
        S_y = int(data[index + 2])
        T_x = int(data[index + 3])
        T_y = int(data[index + 4])
        index += 5
        
        # Calculate the tile indices for start and target
        i_s = S_x // K
        j_s = S_y // K
        k_s = S_y % K if S_y >= 0 else (S_y % K + K) % K
        
        i_t = T_x // K
        j_t = T_y // K
        k_t = T_y % K if T_y >= 0 else (T_y % K + K) % K
        
        # Calculate the number of moves required
        if (i_s == i_t) and (j_s == j_t):
            # Same major tile
            results.append(abs(k_s - k_t))
        else:
            # Different major tiles
            # Calculate the minimum number of moves between major tiles
            delta_i = abs(i_s - i_t)
            delta_j = abs(j_s - j_t)
            
            # If both i and j differences are zero, we only need to move within the same major tile
            if delta_i == 0 and delta_j == 0:
                results.append(abs(k_s - k_t))
            else:
                # Otherwise, we need to move between major tiles
                results.append(delta_i + delta_j)
    
    # Print all results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()