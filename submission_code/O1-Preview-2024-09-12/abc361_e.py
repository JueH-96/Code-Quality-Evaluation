# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]
    total_sum = 0
    for _ in range(N-1):
        A,B,C = map(int, sys.stdin.readline().split())
        adj[A].append((B,C))
        adj[B].append((A,C))
        total_sum += C

    max1_dist = 0
    max1_node = 1

    def dfs1(u, parent, dist):
        nonlocal max1_dist, max1_node
        if dist > max1_dist:
            max1_dist = dist
            max1_node = u
        for v, w in adj[u]:
            if v != parent:
                dfs1(v, u, dist + w)

    dfs1(1, -1, 0)

    max2_dist = 0

    def dfs2(u, parent, dist):
        nonlocal max2_dist
        if dist > max2_dist:
            max2_dist = dist
        for v, w in adj[u]:
            if v != parent:
                dfs2(v, u, dist + w)

    dfs2(max1_node, -1, 0)

    answer = 2 * total_sum - max2_dist
    print(answer)
threading.Thread(target=main).start()