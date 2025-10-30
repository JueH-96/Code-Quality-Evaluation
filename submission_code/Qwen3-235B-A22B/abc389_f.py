import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    L = []
    R = []
    for _ in range(N):
        l, r = map(int, sys.stdin.readline().split())
        L.append(l)
        R.append(r)
    Q = int(sys.stdin.readline())
    queries = [int(sys.stdin.readline()) for _ in range(Q)]
    
    for X in queries:
        v = X
        for i in range(N):
            if L[i] <= v <= R[i]:
                v += 1
        print(v)

threading.Thread(target=main).start()