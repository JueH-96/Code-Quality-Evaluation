# YOUR CODE HERE
MOD = 998244353

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))

    adj = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        adj[i].append((i + 1) % n if (i + 1) % n !=0 else n)
    for u, v in edges:
        adj[u].append(v)

    dp = {}
    def count_paths(curr_node, steps_left):
        if (curr_node, steps_left) in dp:
            return dp[(curr_node, steps_left)]
        if steps_left == 0:
            return 1
        
        count = 0
        for neighbor in adj[curr_node]:
            count = (count + count_paths(neighbor, steps_left - 1)) % MOD
        dp[(curr_node, steps_left)] = count
        return count

    print(count_paths(1, k))

solve()