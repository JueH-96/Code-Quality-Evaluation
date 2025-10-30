# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    S = []
    T = []
    for _ in range(N-1):
        s,t = map(int, sys.stdin.readline().split())
        S.append(s)
        T.append(t)
    DP = [0]*N
    DP[0] = A[0]
    for i in range(N-1):
        mi = DP[i] // S[i]
        transfer = mi * T[i]
        DP[i] = DP[i] - mi * S[i]
        DP[i+1] = A[i+1] + transfer
    print(DP[N-1])
threading.Thread(target=main).start()