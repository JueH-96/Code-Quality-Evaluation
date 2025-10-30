# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    N = len(A)
    X = [0]*(N+1)
    for i in range(N):
        X[i+1] = X[i] ^ A[i]
    max_bit = max(A).bit_length()
    total = 0
    for k in range(max_bit+1):
        count = [0,0]
        total_k = 0
        for i in range(N,-1,-1):
            b = (X[i] >> k) &1
            total_k += count[b ^1]
            count[b] +=1
        total += total_k * (1 << k)
    print(total)


threading.Thread(target=main,).start()