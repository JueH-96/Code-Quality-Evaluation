# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    X = list(map(int, N_and_rest[1:N+1]))
    S_initial = sum(X)
    total_delta_S = 0
    for i in range(N - 3):
        delta_S_i = 2 * (X[i] + X[i+3] - X[i+1] - X[i+2])
        if delta_S_i < 0:
            total_delta_S += delta_S_i
    S_final = S_initial + total_delta_S
    print(S_final)

if __name__ == "__main__":
    threading.Thread(target=main).start()