# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    N = len(A)
    ans = [0]*N
    vis = [False]*N
    for K in range(N):
        if not vis[K]:
            pos_stack = []
            size = A[K]
            positions = [K]
            stack = [K]
            vis[K] = True
            while stack:
                pos = stack.pop()
                for adj in [pos - 1, pos + 1]:
                    if 0 <= adj < N and not vis[adj]:
                        if A[adj] < size:
                            size += A[adj]
                            vis[adj] = True
                            stack.append(adj)
                            positions.append(adj)
            for pos in positions:
                ans[pos] = size
    print(' '.join(map(str, ans)))
threading.Thread(target=main).start()