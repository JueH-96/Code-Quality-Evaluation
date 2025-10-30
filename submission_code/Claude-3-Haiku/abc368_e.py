from collections import defaultdict

def solve():
    N, M, X1 = map(int, input().split())
    trains = []
    for _ in range(M):
        A, B, S, T = map(int, input().split())
        trains.append((A, B, S, T))

    # Sort trains by arrival time
    trains.sort(key=lambda x: x[3])

    # Initialize delays
    delays = [0] * M
    delays[0] = X1

    # Iterate through trains and update delays
    for i in range(1, M):
        for j in range(i):
            if trains[j][1] == trains[i][0] and trains[j][3] <= trains[i][2]:
                delays[i] = max(delays[i], delays[j] + trains[j][3] - trains[i][2])

    print(*delays[1:])

solve()