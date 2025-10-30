def main():
    import sys
    # Read all input tokens.
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    # Parse basic parameters.
    N = int(next(it))
    A = int(next(it))
    B = int(next(it))
    C = int(next(it))
    
    # Read the symmetric distance matrix D.
    # D[i][j] represents the distance between city i and city j.
    D = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            D[i][j] = int(next(it))
    
    # We can travel in two modes:
    #   • Car mode with cost = D[i][j] * A for an edge (i→j).
    #   • Train mode with cost:
    #         - If you switch from car to train: cost = D[i][j] * B + C.
    #         - If already on train: cost = D[i][j] * B.
    #
    # The rule is: you can switch from car mode to train mode (at no extra waiting cost) in a city,
    # but once you switch, you cannot go back.
    #
    # We model the situation as a graph with 2N states:
    #   States 0 .. N-1: being in city i in car mode.
    #   States N .. 2N-1: being in city i in train mode.
    #
    # Transitions:
    #   • From state i (car mode) to state j (car mode):
    #         cost = A * D[i][j]
    #   • From state i (car mode) to state j (train mode):
    #         cost = B * D[i][j] + C   (this is the first train ride after switching)
    #   • From state i (train mode) to state j (train mode):
    #         cost = B * D[i][j]
    #
    # We want the minimum time to reach city N (0-indexed city N-1) in either mode.
    
    tot_states = 2 * N  # Total number of states.
    INF = 10**30  # A very large number.
    dist = [INF] * tot_states
    visited = [False] * tot_states
    
    # Our starting point is city1 in car mode (state 0).
    dist[0] = 0
    
    # We use a simple dense Dijkstra algorithm because the graph is complete.
    # There are tot_states vertices and for each vertex we relax N possible transitions.
    for _ in range(tot_states):
        # Select the unvisited state with the smallest distance.
        u = -1
        best = INF
        for i in range(tot_states):
            if not visited[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u == -1:
            break
        visited[u] = True
        # If u is a car mode state.
        if u < N:
            cur_cost = dist[u]
            # For city u (car mode), relax transitions to every city v.
            # 1. Staying in car mode: cost = current cost + A * D[u][v]
            # 2. Switching to train mode on the move: cost = current cost + B * D[u][v] + C
            Du = D[u]  # Row for city u.
            for v in range(N):
                # Car mode → Car mode transition.
                new_cost = cur_cost + A * Du[v]
                if new_cost < dist[v]:
                    dist[v] = new_cost
                # Car mode → Train mode transition.
                new_cost = cur_cost + B * Du[v] + C
                idx = v + N
                if new_cost < dist[idx]:
                    dist[idx] = new_cost
        else:
            # u is a train mode state.
            city = u - N
            cur_cost = dist[u]
            Du = D[city]
            # From train mode state, only train rides are allowed.
            for v in range(N):
                new_cost = cur_cost + B * Du[v]
                idx = v + N  # Next state must also be in train mode.
                if new_cost < dist[idx]:
                    dist[idx] = new_cost
                    
    # The answer is the minimum cost to reach city N (0-indexed city N-1)
    # in either car mode (state N-1) or train mode (state 2N-1).
    ans = dist[N - 1] if dist[N - 1] < dist[2 * N - 1] else dist[2 * N - 1]
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()