import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    if M == 0:
        # If there are no testimonies, any configuration of confused villagers is valid
        print('0' * N)
        return
    
    # Read testimonies
    testimonies = []
    index = 2
    for _ in range(M):
        A = int(data[index]) - 1
        B = int(data[index + 1]) - 1
        C = int(data[index + 2])
        testimonies.append((A, B, C))
        index += 3
    
    # We need to check for each possible set of confused villagers
    # Since N can be up to 200,000, we cannot check all 2^N possibilities directly.
    # We need a smarter approach, possibly using graph theory or 2-SAT logic.
    
    # We will use a 2-SAT approach where each villager has two possible states:
    # - honest (x_i)
    # - liar (not x_i)
    # We will add implications based on testimonies:
    # - If A says B is honest (C_i = 0), then:
    #   - if A is honest and not confused, B must be honest: x_A -> x_B
    #   - if A is a liar and not confused, B must be a liar: not x_A -> not x_B
    #   - if A is honest and confused, B must be a liar: x_A -> not x_B
    #   - if A is a liar and confused, B must be honest: not x_A -> x_B
    # - If A says B is a liar (C_i = 1), then:
    #   - if A is honest and not confused, B must be a liar: x_A -> not x_B
    #   - if A is a liar and not confused, B must be honest: not x_A -> x_B
    #   - if A is honest and confused, B must be honest: x_A -> x_B
    #   - if A is a liar and confused, B must be a liar: not x_A -> not x_B
    
    # We will use a graph where each villager i has two nodes:
    # - 2*i for x_i (honest)
    # - 2*i+1 for not x_i (liar)
    
    from collections import defaultdict, deque
    
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    
    def add_implication(x, y):
        # x -> y
        graph[x].append(y)
        graph[y^1].append(x^1)
        reverse_graph[y].append(x)
        reverse_graph[x^1].append(y^1)
    
    for A, B, C in testimonies:
        if C == 0:
            # A -> B
            add_implication(2*A, 2*B)     # x_A -> x_B
            add_implication(2*A+1, 2*B+1) # not x_A -> not x_B
            add_implication(2*A, 2*B+1)   # x_A -> not x_B (confused)
            add_implication(2*A+1, 2*B)   # not x_A -> x_B (confused)
        else:
            # A -> not B
            add_implication(2*A, 2*B+1)   # x_A -> not x_B
            add_implication(2*A+1, 2*B)   # not x_A -> x_B
            add_implication(2*A, 2*B)     # x_A -> x_B (confused)
            add_implication(2*A+1, 2*B+1) # not x_A -> not x_B (confused)
    
    # Kosaraju's algorithm to find SCCs and check for contradictions
    def kosaraju_scc(graph, reverse_graph, num_vertices):
        order = []
        visited = [False] * num_vertices
        component = [None] * num_vertices
        
        def dfs1(v):
            visited[v] = True
            for nxt in graph[v]:
                if not visited[nxt]:
                    dfs1(nxt)
            order.append(v)
        
        def dfs2(v, mark):
            component[v] = mark
            for nxt in reverse_graph[v]:
                if component[nxt] is None:
                    dfs2(nxt, mark)
        
        # First pass to get the order of finishing times
        for i in range(num_vertices):
            if not visited[i]:
                dfs1(i)
        
        # Second pass to assign components
        mark = 0
        for v in reversed(order):
            if component[v] is None:
                dfs2(v, mark)
                mark += 1
        
        return component
    
    num_vertices = 2 * N
    scc = kosaraju_scc(graph, reverse_graph, num_vertices)
    
    # Check for contradictions
    for i in range(N):
        if scc[2*i] == scc[2*i+1]:
            print(-1)
            return
    
    # If no contradictions, determine the set of confused villagers
    # We choose the smallest index in each SCC
    answer = ['0'] * N
    for i in range(N):
        if scc[2*i] < scc[2*i+1]:
            answer[i] = '1'  # Choose to be confused if honest implies liar in SCC order
    
    print(''.join(answer))