# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    import bisect
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    P = [list(sys.stdin.readline().strip()) for _ in range(N)]
    total_P = 0
    total_black_cells_in_P = 0
    black_cells = []
    black_in_row = [0]*N
    for u in range(N):
        for v in range(N):
            if P[u][v] == 'B':
                black_cells.append((u, v))
                total_P += 1
                black_in_row[u] +=1

    for _ in range(Q):
        A, B, C, D = map(int, sys.stdin.readline().split())
        base_i = (C - A + 1) // N
        rem_i = (C - A + 1) % N
        base_j = (D - B + 1) // N
        rem_j = (D - B + 1) % N
        Ai = A % N
        Bi = B % N

        K_i = [base_i]*N
        for i in range(rem_i):
            u = (Ai + i)%N
            K_i[u] +=1

        K_j = [base_j]*N
        for i in range(rem_j):
            v = (Bi + i)%N
            K_j[v] +=1

        black_count = 0
        for u,v in black_cells:
            black_count += K_i[u]*K_j[v]
        print(black_count)
threading.Thread(target=main).start()