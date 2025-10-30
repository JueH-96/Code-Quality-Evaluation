# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    tree = [[] for _ in range(N+1)]
    degrees = [0]*(N+1)
    for _ in range(N-1):
        u,v = map(int, sys.stdin.readline().split())
        tree[u].append(v)
        tree[v].append(u)
        degrees[u] +=1
        degrees[v] +=1

    def dfs(node, parent):
        if len(tree[node]) == 1 and node != 1:
            return 0
        max_depth = 0
        for neighbor in tree[node]:
            if neighbor != parent:
                depth = dfs(neighbor, node)
                max_depth = max(max_depth, depth)
        return max_depth +1

    ans = dfs(1, -1)
    print(ans)

threading.Thread(target=main).start()