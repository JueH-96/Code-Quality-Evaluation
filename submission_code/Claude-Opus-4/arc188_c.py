def solve_2sat(n, clauses):
    # Build implication graph
    graph = [[] for _ in range(2 * n)]
    rev_graph = [[] for _ in range(2 * n)]
    
    for a, b in clauses:
        # a -> b means not a OR b
        graph[a ^ 1].append(b)
        graph[b ^ 1].append(a)
        rev_graph[b].append(a ^ 1)
        rev_graph[a].append(b ^ 1)
    
    # Find SCCs using Kosaraju's algorithm
    visited = [False] * (2 * n)
    order = []
    
    def dfs1(v):
        visited[v] = True
        for u in graph[v]:
            if not visited[u]:
                dfs1(u)
        order.append(v)
    
    for i in range(2 * n):
        if not visited[i]:
            dfs1(i)
    
    visited = [False] * (2 * n)
    scc_id = [-1] * (2 * n)
    scc_count = 0
    
    def dfs2(v, id):
        scc_id[v] = id
        for u in rev_graph[v]:
            if scc_id[u] == -1:
                dfs2(u, id)
    
    for v in reversed(order):
        if scc_id[v] == -1:
            dfs2(v, scc_count)
            scc_count += 1
    
    # Check if satisfiable
    for i in range(n):
        if scc_id[2 * i] == scc_id[2 * i + 1]:
            return False
    
    return True

def check_confused(n, testimonies, confused):
    clauses = []
    
    for a, b, c in testimonies:
        a -= 1  # 0-indexed
        b -= 1
        
        if confused[a]:
            # Confused: honest tells lies, liar tells truth
            if c == 0:  # A says B is honest
                # If A is honest (confused), A lies, so B is liar
                # If A is liar (confused), A tells truth, so B is honest
                clauses.append((2 * a, 2 * b + 1))  # x_a -> not x_b
                clauses.append((2 * a + 1, 2 * b))  # not x_a -> x_b
            else:  # A says B is liar
                # If A is honest (confused), A lies, so B is honest
                # If A is liar (confused), A tells truth, so B is liar
                clauses.append((2 * a, 2 * b))  # x_a -> x_b
                clauses.append((2 * a + 1, 2 * b + 1))  # not x_a -> not x_b
        else:
            # Not confused: honest tells truth, liar tells lies
            if c == 0:  # A says B is honest
                # If A is honest, A tells truth, so B is honest
                # If A is liar, A lies, so B is liar
                clauses.append((2 * a, 2 * b))  # x_a -> x_b
                clauses.append((2 * a + 1, 2 * b + 1))  # not x_a -> not x_b
            else:  # A says B is liar
                # If A is honest, A tells truth, so B is liar
                # If A is liar, A lies, so B is honest
                clauses.append((2 * a, 2 * b + 1))  # x_a -> not x_b
                clauses.append((2 * a + 1, 2 * b))  # not x_a -> x_b
    
    return solve_2sat(n, clauses)

n, m = map(int, input().split())
testimonies = []
for _ in range(m):
    a, b, c = map(int, input().split())
    testimonies.append((a, b, c))

# Try all possible confused sets
for mask in range(1 << n):
    confused = [False] * n
    for i in range(n):
        if mask & (1 << i):
            confused[i] = True
    
    if check_confused(n, testimonies, confused):
        result = ''.join('1' if confused[i] else '0' for i in range(n))
        print(result)
        exit()

print(-1)