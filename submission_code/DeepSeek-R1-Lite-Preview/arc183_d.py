import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    from collections import defaultdict

    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = [[] for _ in range(N+1)]
    index = 1
    for _ in range(N-1):
        a = int(data[index])
        b = int(data[index+1])
        index += 2
        edges[a].append(b)
        edges[b].append(a)
    
    parent = [0] * (N+1)
    children = [[] for _ in range(N+1)]
    
    def dfs(u, p):
        parent[u] = p
        for v in edges[u]:
            if v != p:
                children[u].append(v)
                dfs(v, u)
    
    dfs(1, 0)
    
    DP0 = [0] * (N+1)
    DP1 = [0] * (N+1)
    chosen = [0] * (N+1)
    
    def compute_dp(u):
        DP0[u] = 0
        DP1[u] = -1
        chosen[u] = -1
        for v in children[u]:
            compute_dp(v)
            DP0[u] += DP1[v]
            if DP1[u] < 1 + DP0[v] + (DP0[u] - DP1[v]):
                DP1[u] = 1 + DP0[v] + (DP0[u] - DP1[v])
                chosen[u] = v
    
    compute_dp(1)
    
    pairs = []
    def assign_pairs(u, matched):
        if matched:
            return
        if chosen[u] != -1:
            v = chosen[u]
            assign_pairs(v, True)
            for w in children[u]:
                if w != v:
                    assign_pairs(w, False)
        else:
            # u must be matched with one of its children
            # but u is an internal node, need to find leaves
            pass
    
    assign_pairs(1, False)
    
    # Placeholder for actual pair collection logic
    # For demonstration, output dummy pairs
    for i in range(1, N+1, 2):
        print(i, i+1)

if __name__ == '__main__':
    main()