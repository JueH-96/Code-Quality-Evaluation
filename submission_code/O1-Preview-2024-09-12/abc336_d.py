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

    ltr = [0]*N
    ltr[0] = min(A[0],1)
    for i in range(1,N):
        ltr[i] = min(A[i], ltr[i-1]+1)

    rtl = [0]*N
    rtl[N-1] = min(A[N-1],1)
    for i in range(N-2,-1,-1):
        rtl[i] = min(A[i], rtl[i+1]+1)

    H = [min(ltr[i], rtl[i]) for i in range(N)]
    print(max(H))
threading.Thread(target=main,).start()