import sys
import heapq

MOD = 998244353

def power(a, b):
    res = 1
    a %= MOD
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % MOD
        a = (a * a) % MOD
        b //= 2
    return res

def inv(n):
    return power(n, MOD - 2)

def solve():
    N = int(sys.stdin.readline())
    # parents_input[k] is p_{k+1}, parent of vertex k+1 (1-indexed vertices)
    parents_input = list(map(int, sys.stdin.readline().split()))
    # a_input[k] is a_{k+1}, value for vertex k+1 (1-indexed vertices)
    a_input = list(map(int, sys.stdin.readline().split()))

    # Adjacency list for children: adj[u] = list of children of u
    # Vertices are numbered 0 to N.
    adj = [[] for _ in range(N + 1)]
    # parents_input is 0-indexed list for vertices 1 to N.
    # parents_input[i] is parent of vertex i+1.
    for i in range(N): 
        child_node = i + 1 
        parent_node = parents_input[i]
        adj[parent_node].append(child_node)

    # Store a_values. a[k] is the 'a' value for vertex k.
    # Vertices 1 to N have 'a' values. a[0] can be 0 or unused.
    a = [0] * (N + 1) 
    S_a = 0 # This is sum of all a_i for i=1 to N
    # a_input is 0-indexed list for vertices 1 to N.
    # a_input[i] is value for vertex i+1.
    for i in range(N): 
        node_idx = i + 1 
        val = a_input[i]
        a[node_idx] = val
        S_a += val

    # Priority queue stores (-a_v, v_idx) to simulate a max-priority queue.
    # heapq is a min-priority queue by default.
    pq = []

    # Initially, vertex 0 is searched. Its children are eligible.
    for child_of_0 in adj[0]:
        heapq.heappush(pq, (-a[child_of_0], child_of_0))
        
    total_weighted_sum_C = 0
    operations_count = 0 # Counts how many vertices (1..N) have been searched

    while pq:
        # Pop vertex with largest 'a' value (due to -a_v storage).
        # If 'a' values are tied, heapq tie-breaks using v_idx (smaller index preferred).
        neg_a_v, v_idx = heapq.heappop(pq)
        
        operations_count += 1
        
        # Contribution of this vertex to sum: a[v_idx] * operations_count
        # Python handles large intermediate products before taking modulo.
        current_term_value = a[v_idx] * operations_count
        total_weighted_sum_C = (total_weighted_sum_C + current_term_value) % MOD
        
        # Vertex v_idx is now searched. Add its children to the priority queue.
        for child_of_v in adj[v_idx]:
            heapq.heappush(pq, (-a[child_of_v], child_of_v))
            
    # Denominator for expected value is S_a. Compute its modular inverse.
    # S_a % MOD is guaranteed not to be 0 by problem statement.
    S_a_mod = S_a % MOD
    
    ans = (total_weighted_sum_C * inv(S_a_mod)) % MOD
    sys.stdout.write(str(ans) + "
")

# Main execution loop for T test cases
T = int(sys.stdin.readline())
for _ in range(T):
    solve()