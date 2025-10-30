# YOUR CODE HERE
def main():
    import sys
    import threading
    def solve():
        sys.setrecursionlimit(1 << 25)
        N, M = map(int, sys.stdin.readline().split())
        B_r = {}
        W_r = {}
        B_c = {}
        W_c = {}
        for _ in range(M):
            xi, yi, ci = sys.stdin.readline().split()
            xi = int(xi)
            yi = int(yi)
            if ci == 'B':
                B_r[xi] = max(B_r.get(xi, 0), yi)
                B_c[yi] = max(B_c.get(yi, 0), xi)
            else:
                W_r[xi] = min(W_r.get(xi, N+1), yi)
                W_c[yi] = min(W_c.get(yi, N+1), xi)
        rows = set(B_r.keys()) | set(W_r.keys())
        for r in rows:
            B = B_r.get(r, 0)
            W = W_r.get(r, N+1)
            if B >= W:
                print('No')
                return
        cols = set(B_c.keys()) | set(W_c.keys())
        for c in cols:
            B = B_c.get(c, 0)
            W = W_c.get(c, N+1)
            if B >= W:
                print('No')
                return
        print('Yes')
    threading.Thread(target=solve).start()