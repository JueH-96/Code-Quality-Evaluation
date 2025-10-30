import sys

# It is a common practice in competitive programming to increase the recursion
# limit for algorithms like DFS on large graphs.
sys.setrecursionlimit(4 * 2 * 10**5 + 50)

def solve():
    """
    Main function to read input, solve the problem, and print the output.
    """
    try:
        input = sys.stdin.readline
        N, M = map(int, input().split())
        testimonies = [tuple(map(int, input().split())) for _ in range(M)]
    except (IOError, ValueError):
        # Handle empty input or malformed lines
        N, M = 0, 0
        testimonies = []
    
    if N == 0:
        print("")
        return
    if M == 0:
        print("0" * N)
        return

    # 2-SAT formulation
    # Let h_i be 1 if villager i+1 is honest, 0 if liar.
    # Let v_i be 1 if villager i+1 tells the truth, 0 if lies.
    # We know v_i = h_i ^ c_i, where c_i is confusion status.
    # From testimony (A, B, C), we get v_{A-1} = h_{B-1} ^ C.
    
    # We have 2N variables: h_0..h_{N-1}, v_0..v_{N-1}.
    # Variable h_i corresponds to 2-SAT variable i.
    # Variable v_i corresponds to 2-SAT variable N+i.
    
    # 2-SAT variable k is represented by graph nodes:
    # True literal: 2*k
    # False literal: 2*k + 1
    
    num_2sat_vars = 2 * N
    num_nodes = 2 * num_2sat_vars
    adj = [[] for _ in range(num_nodes)]
    rev_adj = [[] for _ in range(num_nodes)]

    def add_implication(u_node, v_node):
        adj[u_node].append(v_node)
        rev_adj[v_node].append(u_node)

    def add_clause(u_literal, v_literal):
        # (u or v) is equivalent to (!u => v) AND (!v => u).
        # A literal's negation is found by XORing with 1.
        add_implication(u_literal ^ 1, v_literal)
        add_implication(v_literal ^ 1, u_literal)

    for A, B, C in testimonies:
        a_idx, b_idx = A - 1, B - 1

        h_b_var_idx = b_idx
        v_a_var_idx = N + a_idx

        h_b_true_literal = 2 * h_b_var_idx
        v_a_true_literal = 2 * v_a_var_idx

        if C == 0:
            # v_a <==> h_b  <=>  (!v_a or h_b) and (v_a or !h_b)
            add_clause(v_a_true_literal ^ 1, h_b_true_literal)
            add_clause(v_a_true_literal, h_b_true_literal ^ 1)
        else: # C == 1
            # v_a <==> !h_b <=>  (!v_a or !h_b) and (v_a or h_b)
            add_clause(v_a_true_literal ^ 1, h_b_true_literal ^ 1)
            add_clause(v_a_true_literal, h_b_true_literal)

    # Kosaraju's algorithm for Strongly Connected Components
    order = []
    visited = [False] * num_nodes
    def dfs1(u):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)

    for i in range(num_nodes):
        if not visited[i]:
            dfs1(i)

    scc = [-1] * num_nodes
    scc_count = 0
    def dfs2(u):
        scc[u] = scc_count
        for v in rev_adj[u]:
            if scc[v] == -1:
                dfs2(v)

    while order:
        u = order.pop()
        if scc[u] == -1:
            dfs2(u)
            scc_count += 1
            
    # Check for contradictions: a variable and its negation in the same SCC
    for i in range(num_2sat_vars):
        if scc[2 * i] == scc[2 * i + 1]:
            print(-1)
            return

    # Construct a valid assignment
    h = [0] * N
    v = [0] * N
    
    for i in range(N):
        # Assign value to h_i (2-SAT variable i)
        h_i_var_idx = i
        h_i_true_literal = 2 * h_i_var_idx
        h_i_false_literal = h_i_true_literal + 1
        if scc[h_i_true_literal] > scc[h_i_false_literal]:
            h[i] = 1

        # Assign value to v_i (2-SAT variable N+i)
        v_i_var_idx = N + i
        v_i_true_literal = 2 * v_i_var_idx
        v_i_false_literal = v_i_true_literal + 1
        if scc[v_i_true_literal] > scc[v_i_false_literal]:
            v[i] = 1
            
    # Calculate confusion string c_i = h_i ^ v_i
    c = [h[i] ^ v[i] for i in range(N)]
    print("".join(map(str, c)))

if __name__ == "__main__":
    solve()