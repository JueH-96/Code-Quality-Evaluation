def main():
    import sys

    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    A = int(input_data[1])
    B = int(input_data[2])
    C = int(input_data[3])

    # Read the distance matrix D (0-based indexing)
    idx = 4
    D = []
    for _ in range(N):
        row = list(map(int, input_data[idx:idx+N]))
        D.append(row)
        idx += N

    INF = 10**18  # A large number for initialization

    # Compute the minimum cost by car from city 1 (index 0) to every city
    dist_car = [INF]*N
    dist_car[0] = 0
    visited = [False]*N
    for _ in range(N):
        u = -1
        best = INF
        for i in range(N):
            if not visited[i] and dist_car[i] < best:
                best = dist_car[i]
                u = i
        if u == -1:
            break
        visited[u] = True
        for v in range(N):
            if not visited[v]:
                alt = dist_car[u] + D[u][v]*A
                if alt < dist_car[v]:
                    dist_car[v] = alt

    # Compute the minimum cost by train from each city to city N (index N-1).
    # We do a similar process, but use city N-1 as the "start".
    dist_train = [INF]*N
    dist_train[N-1] = 0
    visited = [False]*N
    for _ in range(N):
        u = -1
        best = INF
        for i in range(N):
            if not visited[i] and dist_train[i] < best:
                best = dist_train[i]
                u = i
        if u == -1:
            break
        visited[u] = True
        for v in range(N):
            if not visited[v]:
                alt = dist_train[u] + D[u][v]*B + C
                if alt < dist_train[v]:
                    dist_train[v] = alt

    # Now find the minimum of dist_car[k] + dist_train[k] for k=0..N-1.
    # This corresponds to 1->k by car, then k->N by train (possibly k=1 or k=N).
    ans = INF
    for k in range(N):
        cost = dist_car[k] + dist_train[k]
        if cost < ans:
            ans = cost

    print(ans)

# Don't forget to call main()
if __name__ == "__main__":
    main()