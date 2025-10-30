def solve():
    N, M, Q = map(int, input().split())
    ST = []
    for _ in range(M):
        S, T = map(int, input().split())
        ST.append((S, T))
    
    queries = []
    for _ in range(Q):
        L, R = map(int, input().split())
        queries.append((L, R))

    def check_feasibility(people_indices):
        
        def check_person(S, T, weights):
            
            if S < T:
                stamina = 0
                for i in range(S, T):
                    stamina += weights[i-1]
                    if i < T - 1 and stamina <= 0:
                        return False
                if stamina != 0:
                    return False
            else:
                stamina = 0
                for i in range(S - 1, T - 1, -1):
                    stamina += weights[i]
                    if i > T and stamina <= 0:
                        return False
                if stamina != 0:
                    return False
            return True

        import itertools
        
        for weights in itertools.product([-1, 1], repeat=N-1):
            
            feasible = True
            for index in people_indices:
                S, T = ST[index-1]
                if not check_person(S, T, weights):
                    feasible = False
                    break
            
            if feasible:
                return True
        return False

    for L, R in queries:
        people_indices = range(L, R + 1)
        if check_feasibility(people_indices):
            print("Yes")
        else:
            print("No")

solve()