# YOUR CODE HERE
import sys
import sys
import sys
def main():
    import sys
    sys.setrecursionlimit(10000)
    N = int(sys.stdin.readline())
    used_edges = set()
    self_loops = 0
    non_self_edges = []
    edges = []
    for _ in range(N):
        s = sys.stdin.readline().strip()
        if s[0] == s[1]:
            self_loops +=1
        else:
            non_self_edges.append(s)
    # Assign unique ids to non-self-loop edges
    edge_id = {e:i for i,e in enumerate(non_self_edges)}
    M = len(non_self_edges)
    # Build adjacency list
    adj = [[] for _ in range(M)]
    for i, e1 in enumerate(non_self_edges):
        end = e1[1]
        for j, e2 in enumerate(non_self_edges):
            if e1[1] == e2[0]:
                adj[i].append(j)
    # Bipartite matching
    match_to = [-1]*M
    def bpm(u, visited):
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_to[v]==-1 or bpm(match_to[v], visited):
                    match_to[v]=u
                    return True
        return False
    result =0
    for u in range(M):
        visited = [False]*M
        if bpm(u, visited):
            result +=1
    min_paths = (M - result) + self_loops
    print(min_paths)
    
if __name__ == "__main__":
    main()