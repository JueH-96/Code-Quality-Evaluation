# YOUR CODE HERE
import sys
import threading

def main():
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    H = list(map(int, N_and_rest[1:N+1]))
    A_values = [0] * (N+1)
    T_values = [0] * N
    A0 = 0
    current_time = 0
    max_A0_Ai = 0
    A = [0] * (N+1)
    A[0] = 0
    time_needed = 0
    prev_H = 0
    for i in range(N):
        if H[i] > prev_H:
            time_needed += H[i] - prev_H
        time_needed +=1
        T_values[i] = time_needed
        prev_H = H[i]
    print(' '.join(map(str, T_values)))

threading.Thread(target=main).start()