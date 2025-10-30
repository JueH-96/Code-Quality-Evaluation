# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import threading

    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = []
    adj = {}
    edge_used = {}
    nodes = set()
    for _ in range(N):
        s = sys.stdin.readline().strip()
        u = s[0]
        v = s[1]
        edges.append((u, v))
        nodes.add(u)
        nodes.add(v)
        edge_used[(u,v)] = False
        if u not in adj:
            adj[u] = []
        adj[u].append((v))
    path_count = 0
    unused_edges = set(edges)
    def dfs(u, path):
        for v in adj.get(u, []):
            if not edge_used[(u,v)]:
                edge_used[(u,v)] = True
                unused_edges.discard((u,v))
                dfs(v, path)
                return  # After building one path, we return to limit to one path
    while unused_edges:
        for u in nodes:
            for v in adj.get(u, []):
                if not edge_used[(u,v)]:
                    path_count +=1
                    edge_used[(u,v)] = True
                    unused_edges.discard((u,v))
                    dfs(v, [])
                    break
            else:
                continue
            break
    print(path_count)

if __name__ == '__main__':
    threading.Thread(target=main).start()