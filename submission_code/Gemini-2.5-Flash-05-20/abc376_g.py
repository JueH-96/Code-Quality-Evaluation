import sys
import heapq

# Modulo constant
MOD = 998244353

def solve():
    # Read N
    N = int(sys.stdin.readline())

    # Read parent array (p_1, p_2, ..., p_N)
    # parents_input[i-1] stores the parent of vertex i
    parents_input = list(map(int, sys.stdin.readline().split()))

    # Read a_i values (a_1, a_2, ..., a_N)
    # a_values_input[i-1] stores a_i for vertex i
    a_values_input = list(map(int, sys.stdin.readline().split()))

    # Build adjacency list: adj[u] will store a list of children of u
    # Vertices are numbered 0 to N.
    adj = [[] for _ in range(N + 1)]
    for i in range(N):
        # The i-th element of parents_input is p_{i+1}, which is the parent of vertex (i+1).
        parent_of_i_plus_1 = parents_input[i]
        child_vertex = i + 1
        adj[parent_of_i_plus_1].append(child_vertex)

    # Calculate the total sum of a_i values (S_a)
    total_sum_a = sum(a_values_input)

    # Priority queue to store ((-a_value), vertex_id) tuples.
    # We use negative a_value to simulate a max-heap with Python's min-heap (heapq).
    pq = []

    # Initially, vertex 0 is searched. Add its direct children to the priority queue.
    for child_of_root in adj[0]:
        # a_values_input is 0-indexed for a_1 to a_N, so a_values_input[child_vertex - 1]
        heapq.heappush(pq, (-a_values_input[child_of_root - 1], child_of_root))

    # This variable will store the sum (a_v1 * 1 + a_v2 * 2 + ... + a_vN * N)
    # where v_k is the vertex searched at step k according to the greedy strategy.
    expected_weighted_sum = 0

    # operations_count represents 'k' in the formula (a_vk * k). It tracks the number of operations performed.
    operations_count = 0

    # Perform N operations, one for each potential treasure location (vertices 1 to N).
    # The loop continues as long as there are vertices that can be searched.
    # Since all vertices 1 to N are reachable from 0, all N vertices will eventually be processed.
    while pq:
        # Get the vertex with the maximum a_value among available ones
        neg_val_a, u = heapq.heappop(pq)
        val_a = -neg_val_a # Actual a_value for vertex u

        operations_count += 1 # This is the k-th operation

        # Add a_u * k to the total sum
        expected_weighted_sum = (expected_weighted_sum + val_a * operations_count) % MOD

        # Add children of the newly searched vertex 'u' to the priority queue
        for v in adj[u]:
            heapq.heappush(pq, (-a_values_input[v - 1], v))

    # Calculate the modular multiplicative inverse of total_sum_a (S_a^{-1} mod MOD)
    # Using Fermat's Little Theorem: a^(MOD-2) % MOD is inverse of a, since MOD is prime.
    inv_total_sum_a = pow(total_sum_a, MOD - 2, MOD)
    
    # Final answer: (sum(a_vk * k) * S_a^{-1}) % MOD
    ans = (expected_weighted_sum * inv_total_sum_a) % MOD
    
    sys.stdout.write(str(ans) + "
")

# Read the number of test cases
T = int(sys.stdin.readline())

# Solve each test case
for _ in range(T):
    solve()