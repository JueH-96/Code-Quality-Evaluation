def main():
    import math
    
    # Read the number of bottles
    N = int(input().strip())
    
    # Calculate the minimum number of friends needed (bits needed to represent N)
    M = math.ceil(math.log2(N))
    
    # Print the number of friends
    print(M)
    
    # Distribute bottles based on binary representation
    # For friend i, serve all bottles whose (bottle_number - 1) has the i-th bit set to 1
    for i in range(M):
        subset = []
        for b in range(1, N + 1):
            if ((b - 1) >> i) & 1:
                subset.append(b)
        print(len(subset), *subset)
    
    # Read the result string indicating which friends got upset
    S = input().strip()
    
    # Decode the spoiled bottle index from the binary pattern
    spoiled_index = 0
    for i, ch in enumerate(S):
        if ch == '1':
            spoiled_index += (1 << i)
    
    # Convert to 1-based bottle number
    print(spoiled_index + 1)

# Do not forget to call main
if __name__ == "__main__":
    main()