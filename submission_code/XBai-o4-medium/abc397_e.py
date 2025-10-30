import sys
sys.setrecursionlimit(1 << 25)

def main():
    N, K = map(int, sys.stdin.readline().split())
    total_nodes = N * K
    adj = [[] for _ in range(total_nodes + 1)]
    for _ in range(total_nodes - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # Count leaves (degree 1)
    leaves = 0
    for i in range(1, total_nodes + 1):
        if len(adj[i]) == 1:
            leaves += 1
    if leaves > 2 * N:
        print("No")
        return
    
    if K == 1:
        if N == 1:
            print("Yes")
        else:
            print("No")
        return
    elif K == 2:
        if total_nodes % 2 != 0:
            print("No")
            return
        # Check for perfect matching
        matched = [False] * (total_nodes + 1)
        def dfs(u, parent):
            for v in adj[u]:
                if v != parent:
                    if not dfs(v, u):
                        return False
            if parent != 0 and not matched[u]:
                if not matched[parent]:
                    matched[u] = True
                    matched[parent] = True
                else:
                    return False
            return True
        result = dfs(1, 0)
        if result and all(matched[1:]):
            print("Yes")
        else:
            print("No")
    else:
        print("No")

if __name__ == "__main__":
    main()