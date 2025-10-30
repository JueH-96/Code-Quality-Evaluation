def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    S = data[2]
    X = list(map(int, data[3:]))
    
    # Create a list of tuples (position, direction)
    ants = [(X[i], 1 if S[i] == '1' else -1) for i in range(N)]
    
    # Sort ants by their initial positions
    ants.sort()
    
    # Count pairs of ants that will pass each other
    count = 0
    
    # We will use a two-pointer technique to count passing ants
    i = 0
    while i < N:
        j = i + 1
        while j < N and ants[j][0] <= ants[i][0] + 2 * T:
            if ants[i][1] == 1 and ants[j][1] == -1:
                # Ant i is moving right, Ant j is moving left
                if ants[i][0] < ants[j][0]:
                    count += 1
            j += 1
        i += 1
    
    print(count)

if __name__ == "__main__":
    main()