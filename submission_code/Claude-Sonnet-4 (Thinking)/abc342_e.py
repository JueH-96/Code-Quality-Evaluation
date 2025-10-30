def solve():
    N, M = map(int, input().split())
    
    # Parse train information
    train_info = []
    for _ in range(M):
        l, d, k, c, A, B = map(int, input().split())
        train_info.append((l, d, k, c, A, B))
    
    # latest_departure[station] = latest time we can depart from station to reach N
    latest_departure = [-float('inf')] * (N + 1)
    latest_departure[N] = float('inf')  # We can always "depart" from N at any time
    
    # Repeat until convergence
    changed = True
    while changed:
        changed = False
        for l, d, k, c, A, B in train_info:
            if latest_departure[B] == float('inf'):
                # All trains from A to B allow us to reach N
                latest_train_departure = l + (k - 1) * d
                if latest_train_departure > latest_departure[A]:
                    latest_departure[A] = latest_train_departure
                    changed = True
            elif latest_departure[B] > -float('inf'):
                # Find the latest train from A to B that allows us to reach N
                max_departure_from_A = latest_departure[B] - c
                if max_departure_from_A >= l:
                    # There's at least one train from A to B that allows us to reach N
                    i = min(k - 1, (max_departure_from_A - l) // d)
                    latest_train_departure = l + i * d
                    if latest_train_departure > latest_departure[A]:
                        latest_departure[A] = latest_train_departure
                        changed = True
    
    # Output results
    for i in range(1, N):
        if latest_departure[i] == -float('inf'):
            print("Unreachable")
        else:
            print(latest_departure[i])

solve()