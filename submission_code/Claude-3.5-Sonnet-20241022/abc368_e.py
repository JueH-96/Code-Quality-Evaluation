def solve():
    N, M, X1 = map(int, input().split())
    trains = []
    for i in range(M):
        a, b, s, t = map(int, input().split())
        trains.append((a, b, s, t, i+1))
    
    # Initialize delays
    X = [0] * (M+1)
    X[1] = X1
    
    # Keep propagating delays until no more changes
    changed = True
    while changed:
        changed = False
        for i in range(M):
            for j in range(M):
                # Check if transfer is possible from train i to j
                if trains[i][1] == trains[j][0] and trains[i][3] <= trains[j][2]:
                    # Need to ensure delay of j is enough to allow transfer after delay of i
                    required_delay = (trains[i][3] + X[trains[i][4]]) - trains[j][2]
                    if required_delay > X[trains[j][4]]:
                        X[trains[j][4]] = required_delay
                        changed = True
    
    # Print result
    print(*X[2:])

solve()