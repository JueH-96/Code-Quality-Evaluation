import sys
import heapq

def solve():
    # Read input
    N = int(sys.stdin.readline())
    # p is 0-indexed array where p[i] is parent of vertex i+1
    p_input = list(map(int, sys.stdin.readline().split()))
    # a is 0-indexed array where a[i] is weight for vertex i+1
    a_input = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    # Build adjacency list for children: children[u] lists vertices v where p_v = u
    # Vertices are 0 to N. p_input[i] is parent of vertex i+1 for i=0...N-1.
    children = [[] for _ in range(N + 1)]
    for i in range(N):
        vertex_i = i + 1 # Vertex 1-indexed
        parent_of_i = p_input[i] # Parent vertex index (0-indexed or 1-indexed depending on problem)
        # Problem states 0 <= p_i < i for parent of vertex i (1-indexed).
        # This means p_input[i-1] is parent of vertex i.
        # The loop is for i from 0 to N-1, processing vertices 1 to N.
        # Vertex i+1 has parent p_input[i].
        children[p_input[i]].append(vertex_i)

    # Calculate the total sum of a_i for vertices 1..N
    S_actual = sum(a_input)

    # current_sum_a_mod = sum of a_i for vertices not yet searched (among 1..N), modulo MOD
    current_sum_a_mod = sum(a_input) % MOD

    # total_sum_pos_a = sum (a_i * pos(i)) over i=1..N, modulo MOD
    # This is equal to sum (W_{U \setminus C_j}) over j=0..N-1, modulo MOD
    total_sum_pos_a = 0

    # Priority queue stores (-a_i, i) for available vertices (unsearched children of searched vertices)
    # Prioritize by -a_i (max a_i), then by vertex index i (e.g., smaller index for ties, although problem doesn't specify)
    # Using tuple (-a_i, i) naturally sorts by -a_i first, then i.
    pq = []

    # Initially, only vertex 0 is searched. Available vertices are children of 0.
    for child in children[0]:
        # a_input is 0-indexed, a_input[child-1] is the weight for vertex 'child'
        heapq.heappush(pq, (-a_input[child - 1], child))

    # Perform N operations to search vertices 1..N
    for k in range(N):
        # At step k+1 (0-indexed loop variable k, operations are 1-indexed), we search the (k+1)-th vertex.
        # The set of unsearched vertices has total weight current_sum_a_mod.
        # Each of these vertices contributes 1 to their final position count because we perform one more operation.
        # The sum of a_i * pos(i) increases by current_sum_a_mod in this step.
        total_sum_pos_a = (total_sum_pos_a + current_sum_a_mod) % MOD

        # Get vertex with maximum a_i from available vertices
        # This extracts the tuple (-a_v, v) with smallest -a_v (largest a_v)
        if not pq:
             # Should not happen in a valid tree structure where all 1..N are descendants of 0
             break
        neg_a_v, v = heapq.heappop(pq)
        a_v = -neg_a_v # Original a_v value

        # Vertex v is now searched. Update the sum of a_i for remaining unsearched vertices.
        current_sum_a_mod = (current_sum_a_mod - (a_v % MOD) + MOD) % MOD

        # Add children of the newly searched vertex v to the set of available vertices
        for child in children[v]:
             # Add child to PQ
             heapq.heappush(pq, (-a_input[child-1], child))

    # The calculated sum is total_sum_pos_a = sum (a_i * pos(i)) mod MOD
    # The expected value is (sum (a_i * pos(i)) / sum (a_i)) mod MOD
    # This is (total_sum_pos_a * S_actual_inv) mod MOD

    # Calculate the modular multiplicative inverse of S_actual
    # S_actual_inv = S_actual^(MOD-2) mod MOD (by Fermat's Little Theorem)
    S_actual_inv = pow(S_actual, MOD - 2, MOD)

    # Calculate the final expected value modulo MOD
    expected_value = (total_sum_pos_a * S_actual_inv) % MOD

    print(expected_value)


# Read the number of test cases
T = int(sys.stdin.readline())
# Process each test case
for _ in range(T):
    solve()