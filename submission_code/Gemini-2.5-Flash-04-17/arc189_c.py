import sys
from collections import deque

def find_cycles(N, permutation):
    """Find cycles in a permutation graph and map nodes (0 to N-1) to cycle index."""
    visited = [False] * N
    node_to_cycle_idx = [0] * N
    cycle_count = 0
    for i in range(N):
        if not visited[i]:
            cycle_count += 1
            curr = i
            # Traverse the cycle
            while not visited[curr]:
                visited[curr] = True
                node_to_cycle_idx[curr] = cycle_count
                curr = permutation[curr]
            # In a permutation graph, the while loop guarantees that `curr` is the first node
            # revisited in the current traversal, closing the cycle. All nodes traversed
            # belong to this cycle.
    return node_to_cycle_idx

def solve():
    N, X = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    P = list(map(int, sys.stdin.readline().split()))
    Q = list(map(int, sys.stdin.readline().split()))

    # Adjust X and permutations to be 0-indexed
    X -= 1
    P = [p - 1 for p in P]
    Q = [q - 1 for q in Q]

    # Step 1: Check if balls from initially dirty boxes can theoretically reach X via P and Q cycles.
    # A ball starting at i != X can reach X via P steps iff i and X are on the same P-cycle.
    # Same for blue balls and Q steps.
    p_cycle_ids = find_cycles(N, P)
    q_cycle_ids = find_cycles(N, Q)

    x_p_cycle_id = p_cycle_ids[X]
    x_q_cycle_id = q_cycle_ids[X]

    for i in range(N):
        if i == X:
            continue
        if A[i] > 0:
            # Box i has red balls initially. Must reach X via P.
            # This is possible iff i is in the same P-cycle as X.
            if p_cycle_ids[i] != x_p_cycle_id:
                print(-1)
                return
        if B[i] > 0:
            # Box i has blue balls initially. Must reach X via Q.
            # This is possible iff i is in the same Q-cycle as X.
            if q_cycle_ids[i] != x_q_cycle_id:
                print(-1)
                return

    # Step 2: Find the set of boxes (other than X) that must be operated on.
    # A box i != X must be operated on if it initially contains balls (that can reach X),
    # or if it receives balls from an operation on another box j != X.
    # This forms a dependency: operating on j != X might require operating on P[j] != X
    # and Q[j] != X.
    # The set of boxes that must be operated on is the set of nodes reachable from
    # the initial set of dirty boxes (that passed condition 1) in graph G'
    # where nodes are {0...N-1} \ {X} and edges are j -> P[j] (if P[j]!=X) and j -> Q[j] (if Q[j]!=X).

    # Build adjacency list for G' (edges represent potential flow of balls requiring operation)
    # Nodes in G' are {0..N-1} excluding X.
    adj_g_prime = [[] for _ in range(N)]
    for j in range(N):
        if j == X:
            continue
        # If operation on j moves red balls to P[j] and P[j] is not X, P[j] might need operation.
        if P[j] != X:
            adj_g_prime[j].append(P[j])
        # If operation on j moves blue balls to Q[j] and Q[j] is not X, Q[j] might need operation.
        if Q[j] != X:
            adj_g_prime[j].append(Q[j])

    # Use BFS to find reachable nodes from initial dirty boxes in G'
    must_operate = [False] * N
    q = deque()

    # Add initial dirty boxes (excluding X) to the queue and mark as must_operate
    # Only add if initially dirty AND passed the cycle condition check above.
    # The initial check already filtered based on cycles, so just check A[i]>0 or B[i]>0.
    for i in range(N):
        if i == X:
            continue
        if A[i] > 0 or B[i] > 0:
            # If i contains balls initially, it must be operated on to move them.
            if not must_operate[i]:
                must_operate[i] = True
                q.append(i)

    # If no boxes initially contain balls outside X, 0 operations needed.
    # The BFS will correctly result in operated_count = 0.

    while q:
        u = q.popleft()

        # Explore neighbors in G'. These are boxes that receive balls from u's operation.
        for v in adj_g_prime[u]:
            # v is guaranteed to be != X by construction of adj_g_prime
            if not must_operate[v]:
                must_operate[v] = True
                q.append(v)

    # The minimum number of operations is the count of unique boxes (other than X)
    # that were marked as must_operate. Each such box must be operated on at least once.
    operated_count = 0
    for i in range(N):
        if i != X and must_operate[i]:
             operated_count += 1

    print(operated_count)

solve()