# YOUR CODE HERE
import sys
import threading
sys.setrecursionlimit(1 << 25)
def main():
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N)]
    for _ in range(N - 1):
        A, B = map(int, sys.stdin.readline().split())
        A -= 1  # convert to 0-based index
        B -= 1
        adj[A].append(B)
        adj[B].append(A)
    C = list(map(int, sys.stdin.readline().split()))
    TotalC = sum(C)

    sumC = [0] * N  # sum of C_i in subtree rooted at u
    dp = [0] * N    # total weighted distance from u to nodes in its subtree

    def dfs1(u, parent):
        sumC[u] = C[u]
        dp_u = 0
        for v in adj[u]:
            if v != parent:
                dfs1(v, u)
                sumC[u] += sumC[v]
                dp_u += dp[v] + sumC[v]
        dp[u] = dp_u

    dfs1(0, -1)  # root at node 0

    f = [0] * N
    f[0] = dp[0]
    min_f = f[0]

    def dfs2(u, parent):
        nonlocal min_f
        for v in adj[u]:
            if v != parent:
                f_v = f[u] + TotalC - 2 * sumC[v]
                f[v] = f_v
                min_f = min(min_f, f_v)
                dfs2(v, u)

    dfs2(0, -1)

    # Since the answer could be negative, we need to ensure it's an integer
    print(int(min_f))

if __name__ == '__main__':
    threading.Thread(target=main).start()