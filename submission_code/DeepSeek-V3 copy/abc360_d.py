def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    S = data[2]
    X = list(map(int, data[3:3+N]))
    
    # Create a list of tuples (position, direction, index)
    ants = []
    for i in range(N):
        direction = 1 if S[i] == '1' else -1
        ants.append((X[i], direction, i+1))
    
    # Sort ants by position
    ants.sort()
    
    # Initialize the count of passing pairs
    count = 0
    
    # Iterate through all pairs of ants
    for i in range(N):
        for j in range(i+1, N):
            x1, d1, idx1 = ants[i]
            x2, d2, idx2 = ants[j]
            # If one is moving right and the other is moving left
            if d1 == 1 and d2 == -1:
                # Calculate the time when they meet
                distance = x2 - x1
                if distance == 0:
                    # They are at the same position initially, but since all positions are distinct, this should not happen
                    pass
                else:
                    time = distance / 2
                    if 0 <= time <= T + 0.1:
                        count += 1
    
    print(count)

if __name__ == "__main__":
    main()