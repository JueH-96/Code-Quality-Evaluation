# YOUR CODE HERE

import sys

def solve():
    N, X, Y = map(int, input().split())
    bus_stops = [list(map(int, input().split())) for _ in range(N-1)]
    Q = int(input())
    queries = [int(input()) for _ in range(Q)]

    bus_times = [0] * N
    for i in range(N-1):
        P, T = bus_stops[i]
        for j in range(P, N, P):
            bus_times[j] = T

    for q in queries:
        time = X + Y
        for i in range(N-1, 0, -1):
            time = min((time + P - 1) // P * P, bus_times[i]) + T
        time += X
        print(time)

solve()