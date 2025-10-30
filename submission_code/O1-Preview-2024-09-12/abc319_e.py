# YOUR CODE HERE
import sys
import threading

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    N, X, Y = map(int, sys.stdin.readline().split())
    N = int(N)
    X = int(X)
    Y = int(Y)
    P = []
    T = []
    for _ in range(N - 1):
        Pi, Ti = map(int, sys.stdin.readline().split())
        P.append(int(Pi))
        T.append(int(Ti))

    Q = int(sys.stdin.readline())
    q_list = [int(sys.stdin.readline()) for _ in range(Q)]

    # Precompute duration[t_mod]
    MAX_P = 8
    LCM = 840  # lcm of numbers from 1 to 8
    duration = [0] * LCM
    for t_mod in range(LCM):
        t = t_mod
        t += X  # Walk to bus stop 1
        for i in range(N - 1):
            wait_i = (-t) % P[i]
            t += wait_i + T[i]
        t += Y  # Walk to Aoki's house
        duration[t_mod] = t - t_mod

    for q_i in q_list:
        t_mod = q_i % LCM
        arrival_time = q_i + duration[t_mod]
        print(arrival_time)

def threading_main():
    threading.Thread(target=main).start()

threading_main()