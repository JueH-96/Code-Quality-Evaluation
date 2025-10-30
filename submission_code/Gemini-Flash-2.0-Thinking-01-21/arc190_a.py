import sys
import collections

# Helper function to find the index of the largest critical point <= x
# y is the sorted list of unique critical points, including 0 and N+1
def get_critical_point_index(y, x):
    # Use bisect_right to find the index of the first element > x
    # The index of the largest element <= x is one less.
    # Handle the case where x is smaller than the smallest critical point (>0).
    # y starts with 0.
    from bisect import bisect_right
    if x < y[0]:
        return -1 # Should not happen with y[0]=0
    if x >= y[-1]:
        return len(y) - 1 # Should not happen if y includes N+1 and we look up to N

    # Find the index of the first element in y strictly greater than x
    idx = bisect_right(y, x)
    # The largest element <= x is y[idx-1]
    return idx - 1

def solve():
    N, M = map(int, sys.stdin.readline().split())
    ops = []
    critical_points_set = {0, N + 1}
    for i in range(M):
        L, R = map(int, sys.stdin.readline().split())
        ops.append((L, R))
        critical_points_set.add(L)
        critical_points_set.add(R + 1)

    y = sorted(list(critical_points_set))
    m = len(y)

    # Map critical point values to indices
    y_to_idx = {point: idx for idx, point in enumerate(y)}

    # dp[j]: min cost to cover interval [1, y[j]]
    # Node j in DP graph represents covering [1, y[j]]
    dp = [float('inf')] * m
    parent = [None] * m
    parent_op_type = [None] * m
    parent_op_idx = [None] * m

    # Base case: covering [1, y[0]] = [1, 0] (empty interval) costs 0
    dp[0] = 0

    # Store operations grouped by L, R, and complement ranges for faster queries
    ops_by_l = collections.defaultdict(list)
    ops_by_r = collections.defaultdict(list)
    for i, (L, R) in enumerate(ops):
        ops_by_l[L].append((R, i))
        ops_by_r[R].append((L, i))

    # Precompute max R for operations starting <= point p
    # max_reach[p] = max {R_i | L_i <= p}
    max_reach = [-1] * (N + 2)
    current_max_r = -1
    op_indices_at_r = collections.defaultdict(list) # Store which op achieves max_reach
    current_op_at_max_r = -1

    op_list_sorted_by_l = sorted([(L, R, i) for i, (L, R) in enumerate(ops)])
    op_idx = 0
    for p in range(1, N + 2):
        # Add operations that start at p-1
        while op_idx < M and op_list_sorted_by_l[op_idx][0] <= p -1:
             L, R, i = op_list_sorted_by_l[op_idx]
             if R > current_max_r:
                 current_max_r = R
                 current_op_at_max_r = i
             op_idx += 1
        max_reach[p] = current_max_r
        op_indices_at_r[p] = current_op_at_max_r


    # Iterate through the critical points y[j] from left to right
    for j in range(1, m):
        # y[j] is the current right boundary point. We want to cover [1, y[j]].
        # This requires covering [1, y[j-1]] (cost dp[j-1]) and extending coverage to include y[j].

        # Option 1: Cover y[j] using an Operation 2
        # An Op 2 on [L_i, R_i] covers y[j] if y[j] < L_i or y[j] > R_i.
        # If y[j] < L_i, it covers [1, L_i-1]. If L_i-1 >= y[j], this covers [1, y[j]]. This means L_i > y[j].
        # If y[j] > R_i, it covers [R_i+1, N]. This covers [1, y[j]] if R_i+1 <= 1, i.e. R_i < 1. (Not useful for y[j] >= 1)
        # So, Op 2 covers [1, y[j]] effectively if L_i > y[j].
        # The cheapest way to potentially cover [1, y[j]] using an *additional* Op 2 is if *any* Op 2 can cover y[j].
        # If any Op 2 covers y[j], it means the intersection of *all* selected Op 2 ranges does *not* contain y[j].
        # This seems to lead to the shortest path formulation directly.

        # Consider transition from dp[k] (covering [1, y[k]]) to dp[j] (covering [1, y[j]]) for k < j.
        # This transition costs 1 if there exists a single operation (type 1 or 2) that covers the interval [y[k]+1, y[j]].
        # [y[k]+1, y[j]] is covered by Op 1 on [L_i, R_i] if L_i <= y[k]+1 and R_i >= y[j].
        # [y[k]+1, y[j]] is covered by Op 2 on [L_i, R_i] if y[j] < L_i or y[k]+1 > R_i.

        # To optimize: iterate k from j-1 down to 0.
        # For fixed j and decreasing k:
        # Check Op 2 covering [y[k]+1, y[j]]: y[j] < L_i (constant check for fixed j) or y[k]+1 > R_i (condition becomes stricter for decreasing k).
        # Check Op 1 covering [y[k]+1, y[j]]: L_i <= y[k]+1 (condition becomes stricter for decreasing k) and R_i >= y[j] (constant check for fixed j).

        # Let's try optimizing the transitions for dp[j]
        # dp[j] = min_{0 <= k < j} (dp[k] + 1 if edge (k, j) exists)

        # Calculate min L_i for L_i > y[j] (constant for fixed j)
        min_l_gt_yj = N + 2
        for L, R_list in ops_by_l.items():
            if L > y[j]:
                min_l_gt_yj = min(min_l_gt_yj, L)

        # Calculate max R_i for R_i < y[j] (constant for fixed j)
        max_r_lt_yj = -1
        for R, L_list in ops_by_r.items():
            if R < y[j]:
                max_r_lt_yj = max(max_r_lt_yj, R)

        # Iterate backwards from k = j-1 down to 0
        # The Op 2 condition "y[k]+1 > R_i" is equivalent to "R_i < y[k]+1".
        # We need to efficiently check if exists i: R_i < y[k]+1. This is equivalent to checking if max R_i overall is < y[k]+1.
        # But we need to find an operation that covers the interval, not just check if *any* R_i < y[k]+1.
        # The condition is that *one* op 2 covers the interval [y[k]+1, y[j]].

        # Simplified edge check:
        # Edge (k, j) exists if there is *some* operation i that covers [y[k]+1, y[j]] with type 1 or type 2.
        # Type 1: [L_i, R_i] covers [y[k]+1, y[j]] => L_i <= y[k]+1 AND R_i >= y[j].
        # Type 2: [1, L_i-1] U [R_i+1, N] covers [y[k]+1, y[j]] => [y[k]+1, y[j]] <= [1, L_i-1] OR [y[k]+1, y[j]] <= [R_i+1, N].
        # [y[k]+1, y[j]] <= [1, L_i-1] => y[j] < L_i.
        # [y[k]+1, y[j]] <= [R_i+1, N] => y[k]+1 > R_i.
        # So edge (k, j) exists if EXISTS i: (L_i <= y[k]+1 AND R_i >= y[j]) OR (y[j] < L_i OR y[k]+1 > R_i).

        # Iterate k from j-1 down to 0
        # Need to find min(dp[k] + 1) over k where edge (k, j) exists.

        # Optimize edge checks during DP calculation for dp[j]
        # For fixed j, iterate k from j-1 down to 0.
        # Maintain max R_i for L_i <= y[k]+1 as k decreases.
        # Maintain max R_i for R_i < y[k]+1 as k decreases.
        # Maintain min L_i for L_i > y[j].

        # min_cost_op1_to_reach_j[k] = min_{i: L_i <= y[k]+1, R_i >= y[j]} (dp[k] + 1).
        # This is not correct. The DP is on critical points, not arbitrary points.

        # Correct DP state and transitions using critical points as nodes in a graph:
        # Nodes are indices of critical points $0, \ldots, m-1$. Node $k$ represents covering $[1, y_k]$.
        # Edge from node $k$ to node $j$ ($k < j$) with weight 1 if we can cover $[y_k+1, y_j]$ using a single operation.
        # $[y_k+1, y_j]$ is covered by Op 1 on $[L_i, R_i]$ if $L_i \le y_k+1$ AND $R_i \ge y_j$.
        # $[y_k+1, y_j]$ is covered by Op 2 on [L_i, R_i] if $y_j < L_i$ OR $y_k+1 > R_i$.
        # $dp[j]$ = shortest path distance from node 0 to node j.

        # Dijkstra-like approach or Bellman-Ford like DP on DAG
        # $dp[j] = \min_{0 \le k < j} (dp[k] + \text{cost}(k, j))$ where cost(k, j) = 1 if edge exists, infinity otherwise.
        # This is O(m^2) if edge check is O(1). Edge check takes O(M). Total O(m^2 M).

        # Optimize edge check for all (k, j) pairs.
        # For each operation i, it potentially creates edges:
        # Op 1 [L_i, R_i]: Edge (k, j) if $y_k+1 \ge L_i$ and $y_j \le R_i$. Find $k_{max}$ s.t. $y_{k_{max}}+1 \le L_i$ and $j_{min}$ s.t. $y_{j_{min}} \ge R_i$.
        #   $k_{max} = \text{idx}(L_i - 1)$. $j_{min} = \text{idx}(R_i)$? No, $y_j$ is the right boundary. Need $y_j \le R_i$. Smallest $j$ s.t. $y_j \ge R_i$.
        #   Smallest index $j_{min}$ such that $y_{j_{min}} \ge R_i$. Use bisect_left.
        #   Largest index $k_{max}$ such that $y_{k_{max}}+1 \le L_i$. Equivalent to $y_{k_{max}} \le L_i-1$. $k_{max} = \text{idx}(L_i-1)$.
        #   So Op 1 [L_i, R_i] potentially adds edges $(k, j)$ for $0 \le k \le k_{max}$ and $j_{min} \le j < m$.
        # Op 2 [L_i, R_i]: Edge (k, j) if $y_j < L_i$ OR $y_k+1 > R_i$.
        #   $y_j < L_i$: Smallest $j$ s.t. $y_j \ge L_i$. $j_{min} = \text{bisect_left}(y, L_i)$. For $j \ge j_{min}$, this condition holds. Any $k < j_{min}$ can have an edge to $j$.
        #   $y_k+1 > R_i$: Largest $k$ s.t. $y_k+1 > R_i$. Equivalent to $y_k \ge R_i$. Smallest $k$ s.t. $y_k \ge R_i$ is $k_{min} = \text{bisect_left}(y, R_i)$. For $k < k_{min}$, condition holds. Edge (k, j) exists for $k < k_{min}$ and any $j > k$.

        # The DP can be written as:
        # $dp[j] = \min ($
        #   $\min_{k<j \text{ s.t. } \exists i: L_i \le y_k+1, R_i \ge y_j} (dp[k] + 1),$
        #   $\min_{k<j \text{ s.t. } \exists i: y_j < L_i} (dp[k] + 1),$
        #   $\min_{k<j \text{ s.t. } \exists i: y_k+1 > R_i} (dp[k] + 1)$
        # )

        # $dp[j] = \min ($
        #   $\min \{ dp[k] + 1 \mid k < j, \exists i: L_i \le y_k+1, R_i \ge y_j \},$
        #   $\min \{ dp[k] + 1 \mid k < j, \exists i: y_j < L_i \text{ (let } i_0 \text{ be one such op)} \},$
        #   $\min \{ dp[k] + 1 \mid k < j, \exists i: y_k+1 > R_i \text{ (let } i_1 \text{ be one such op)} \}$
        # )

        # $\min \{ dp[k] + 1 \mid k < j, \exists i: y_j < L_i \}$: If there is any op $i$ with $L_i > y_j$, then for any $k < j$, using this op $i$ as type 2 covers $[1, L_i-1]$. If $L_i-1 \ge y_j$, this covers $[1, y_j]$. Cost $dp[k] + 1$? No, the Op 2 covers a range, not an interval based on $k$.

        # Let $dp[j]$ be min cost to cover $[1, y_j]$.
        # $dp[j]$ comes from extending coverage from some $y_k$.
        # The transition involves covering $[y_k+1, y_j]$.
        # $dp[j] = \min_{0 \le k < j} (dp[k] + \text{cost to cover } [y_k+1, y_j])$.

        # Cost to cover $[y_k+1, y_j]$ using *any* set of ops.
        # This is complex. The shortest path formulation with cost 1 edge is likely correct. Need faster edge check.

        # Edge $(k, j)$ exists if EXISTS i: (L_i <= y[k]+1 AND R_i >= y[j]) OR (y[j] < L_i OR y[k]+1 > R_i).
        # For fixed j, iterate k from j-1 down to 0.
        # Check Op 2: $\exists i: y_j < L_i$ (constant for fixed j) OR $\exists i: R_i < y[k]+1$.
        # Min L_i > y[j] is min(L for L,R in ops if L > y[j]). Let this be min_L_gt_yj.
        # Max R_i < y[k]+1 is max(R for L,R in ops if R < y[k]+1). Let this be max_R_lt_yk1.
        # Op 2 condition: min_L_gt_yj <= N+1 OR max_R_lt_yk1 >= 1.

        # Optimized DP step for dp[j]:
        # $dp[j] = \min(dp[j], dp[j-1] + cost)$
        # $dp[j] = \min_{0 \le k < j} (dp[k] + 1 \text{ if edge } (k, j) \text{ exists})$.

        # Optimization: For fixed j, update dp[j] from dp[k] for k < j.
        # Transitions to j:
        # 1. From k via Op 1: $L_i \le y_k+1, R_i \ge y_j$. $dp[j] = \min(dp[j], dp[k]+1)$.
        # 2. From k via Op 2 ( $y_j < L_i$ ): $dp[j] = \min(dp[j], dp[k]+1)$ if $\exists i: L_i > y_j$.
        # 3. From k via Op 2 ( $y_k+1 > R_i$ ): $dp[j] = \min(dp[j], dp[k]+1)$ if $\exists i: R_i < y_k+1$.

        # For fixed $j$:
        # $dp[j] = \min(dp[j], \min_{k<j, \exists i: L_i \le y_k+1, R_i \ge y_j} (dp[k]+1))$. (Type 1 transitions)
        # If $\exists i: L_i > y_j$: $dp[j] = \min(dp[j], \min_{k<j} (dp[k]+1))$. This seems wrong.

        # Correct logic for Op 2 transition: If we decide to use an Op 2 $i$ with $L_i > y_j$, this covers $[1, L_i-1]$, including $[1, y_j]$. This one Op 2 is sufficient to cover $[1, y_j]$ by itself (assuming $L_i-1 \ge y_j$, which holds if $L_i > y_j$). Cost 1. The previous state doesn't matter as much?

        # Let's compute $dp[j]$ based on options to cover $y_j$.
        # $y_j$ must be covered by some Op 1 or Op 2.
        # Option A: $y_j$ is covered by Op 1 on $i$ ($L_i \le y_j \le R_i$). If $R_i \ge y_j$, this Op 1 covers $[L_i, y_j]$. Need $[1, L_i-1]$ covered. Cost $dp[\text{idx}(L_i-1)] + 1$.
        # $dp[j] = \min_{i: L_i \le y_j, R_i \ge y_j} (dp[\text{idx}(L_i-1)] + 1)$.
        # Option B: $y_j$ is covered by Op 2 on $i$ ($y_j < L_i$ or $y_j > R_i$).
        # If $y_j < L_i$: Op 2 covers $[1, L_i-1]$. If $L_i-1 \ge y_j$, this covers $[1, y_j]$.
        # If $y_j > R_i$: Op 2 covers $[R_i+1, N]$.
        # If we use an Op 2 $i$ with $L_i > y_j$: it covers $[1, L_i-1]$, sufficient for $[1, y_j]$. Cost 1.
        # This suggests $dp[j] = \min(dp[j], 1)$ if there is any op with $L_i > y_j$. No, this would be the cost if we only use one Op 2.

        # $dp[j]$ min cost to cover $[1, y_j]$.
        # $dp[j] = \min($
        #   $dp[j-1] + \text{cost to cover } [y_{j-1}+1, y_j]$ (by ops not necessarily covering points < $y_{j-1}+1$)
        #   $\min_{i: L_i \le y_j \le R_i} (dp[\text{idx}(L_i-1)] + 1)$
        # )
        # Cost to cover $[y_{j-1}+1, y_j]$ by Op 2: if $\exists i: y_j < L_i$ or $y_{j-1}+1 > R_i$. Cost 1.

        # Let $dp[j]$ be min cost to cover $[1, y_j]$.
        # $dp[0] = 0$.
        # For $j=1 \ldots m-1$:
        # $dp[j] = \infty$.
        # Option 1: Use an Op 1 $i$ with $L_i \le y_j \le R_i$. This Op covers $[L_i, R_i]$. If $R_i \ge y_j$, it covers $[L_i, y_j]$.
        # We need $[1, L_i-1]$ covered. Cost $dp[\text{idx}(L_i-1)] + 1$.
        # Iterate over all ops $i$ with $L_i \le y_j$ and $R_i \ge y_j$. Update $dp[j]$.
        # Option 2: Use an Op 2 $i$ with $y_j < L_i$ or $y_j > R_i$.
        # If $y_j < L_i$: Op 2 covers $[1, L_i-1]$. If $L_i-1 \ge y_j$, covers $[1, y_j]$. Cost $dp[j-1] + 1$.
        # If $y_j > R_i$: Op 2 covers $[R_i+1, N]$. This doesn't directly help cover $[1, y_j]$.

        # $dp[j] = dp[j-1]$. # What does this mean? It means covering [1, y[j-1]] and y[j] is already covered?
        # It must be $dp[j] = \min($
        #    $dp[j-1] + \text{cost to cover } y_j \text{ using Op 2 that covers } y_j \text{ and possibly previous points}$,
        #    $\min_{i: L_i \le y_j \le R_i} (dp[\text{idx}(L_i-1)] + 1)$
        # )
        # The first term could be $\min_{i: y_j < L_i \text{ or } y_j > R_i} (dp[\text{some index}] + 1)$. If $y_j < L_i$, Op 2 covers $[1, L_i-1]$. Index $\text{idx}(L_i-1)$. Cost $dp[\text{idx}(L_i-1)]+1$.
        # If $y_j > R_i$, Op 2 covers $[R_i+1, N]$. Does this help cover $[1, y_j]$? Only if $R_i+1 \le 1$, i.e. $R_i \le 0$.

        # $dp[j] = \min_{k < j} (dp[k] + \text{cost to cover } [y_k+1, y_j])$.
        # Cost to cover $[y_k+1, y_j]$ is 1 if exists single op $i$ covering it.
        # Edge $(k, j)$ if $\exists i: (L_i \le y_k+1, R_i \ge y_j) \lor (y_j < L_i \lor y_k+1 > R_i)$.
        # DP with $O(m^2)$ states, each update $O(1)$ using precomputed reachability.

        # Precomputation for $O(m^2)$ DP:
        # For each pair $(k, j)$ with $k < j$: Does edge $(k, j)$ exist?
        # E1(k, j): $\exists i: L_i \le y_k+1, R_i \ge y_j$.
        # E2(k, j): $\exists i: y_j < L_i$ or $y_k+1 > R_i$.
        # Build adjacency matrix or lists for critical point nodes.
        adj = [[] for _ in range(m)]
        op_info = {} # Stores which op creates the edge

        # E1 edges: For each op i=(L_i, R_i), find max k s.t. y[k]+1 <= L_i and min j s.t. y[j] >= R_i.
        # $k_{max\_idx} = \text{idx}(L_i-1)$. $j_{min\_idx} = \text{bisect_left}(y, R_i)$.
        # If $k_{max\_idx} >= 0$ and $j_{min\_idx} < m$ and $y[j_{min\_idx}] >= R_i$ and $y[k_{max\_idx}]+1 <= L_i$:
        # Op i creates edges $(k, j)$ for $0 \le k \le k_{max\_idx}$ and $j_{min\_idx} \le j < m$.
        # This seems too many edges.

        # Let's use the DP state and transitions again.
        # $dp[j]$: min cost to cover $[1, y_j]$.
        # $dp[0]=0$.
        # For $j=1 \dots m-1$:
        # $dp[j] = \infty$.
        # Option 1: Cover $y_j$ using an Op 1 $i$ with $L_i \le y_j \le R_i$. If $R_i \ge y_j$, covers $[L_i, y_j]$. Need $[1, L_i-1]$ covered.
        # Possible transitions via Op 1: from $dp[\text{idx}(L_i-1)]$ to $dp[j]$ if $L_i \le y_j$ and $R_i \ge y_j$. Cost 1.
        # Option 2: Cover $y_j$ using an Op 2 $i$ with $y_j < L_i$ or $y_j > R_i$. If $y_j < L_i$, covers $[1, L_i-1]$. If $L_i-1 \ge y_j$, covers $[1, y_j]$.
        # Possible transitions via Op 2 (y_j < L_i): from $dp[\text{idx}(L_i-1)]$ to $dp[j]$ if $y_j < L_i$. Cost 1.

        # $dp[j] = \min ($
        #   $\min_{i: L_i \le y_j, R_i \ge y_j} (dp[\text{idx}(L_i-1)] + 1)$,  # Using Op 1 covering y_j
        #   $\min_{i: y_j < L_i \text{ or } y_j > R_i} (dp[\text{idx}(L_i-1) \text{ or something else?}] + 1)$ # Using Op 2 covering y_j
        # )

        # Let's follow sample output logic. Op 1 type 2, Op 3 type 1.
        # Op 1 (2,4) type 2 covers [1,1] U [5,5]. Op 3 (1,4) type 1 covers [1,4].
        # DP state needs to capture which operations are already used/committed? No.

        # Final attempt at DP state: $dp[j]$ min cost to cover $[1, y_j]$.
        # $dp[j]$ comes from some previous state $dp[k]$ ($k < j$). The interval $[y_k+1, y_j]$ is covered.
        # The interval $[y_k+1, y_j]$ must be covered by some operation(s) *used for this step*.
        # Option A: Covered by a single Op 1 $i$: $L_i \le y_k+1, R_i \ge y_j$. Cost $dp[k] + 1$.
        # Option B: Covered by a single Op 2 $i$: $y_j < L_i$ or $y_k+1 > R_i$. Cost $dp[k] + 1$.

        # This is the shortest path on DAG. $dp[j] = \min_{0 \le k < j, \text{edge}(k, j)} (dp[k] + 1)$.
        # Edge check: $\exists i: (L_i \le y_k+1, R_i \ge y_j) \lor (y_j < L_i \lor y_k+1 > R_i)$.
        # Let's build the graph explicitly. Nodes $0, \ldots, m-1$.
        # $dp = [float('inf')] * m$. $dp[0] = 0$.
        # parent = [None] * m
        # parent_op_idx = [None] * m
        # parent_op_type = [None] * m

        # Adjacency list: adj[k] = list of (j, op_idx, op_type)
        adj = [[] for _ in range(m)]

        for i, (L, R) in enumerate(ops):
            # Op 1: covers [L, R]. Can cover [y_k+1, y_j] if $L \le y_k+1$ and $R \ge y_j$.
            # Find max k s.t. y_k+1 <= L => y_k <= L-1. $k_{max} = \text{idx}(L-1)$.
            # Find min j s.t. y_j >= R. $j_{min} = \text{bisect_left}(y, R)$.
            k_max = get_critical_point_index(y, L - 1)
            j_min = bisect_left(y, R)
            if j_min < m and y[j_min] >= R: # Check if R is actually covered by some y_j
                 for k in range(k_max + 1):
                     for j in range(j_min, m):
                        # Add edge (k, j) with cost 1, type 1, op i
                        adj[k].append((j, i, 1))

            # Op 2: covers [1, L-1] U [R+1, N]. Can cover [y_k+1, y_j] if $y_j < L$ OR $y_k+1 > R$.
            # Case 1: y_j < L. Find min j s.t. y_j >= L. $j_{min} = \text{bisect_left}(y, L)$.
            # For any j from $j_{min}$ to $m-1$, y_j >= L > y_j for k < j. So y_j < L.
            # For j from $j_{min}$ to $m-1$, and any k < j: Op 2 covers [y_k+1, y_j].
            j_min_op2_L = bisect_left(y, L)
            for j in range(j_min_op2_L, m):
                for k in range(j):
                     # Add edge (k, j) with cost 1, type 2, op i
                     adj[k].append((j, i, 2))

            # Case 2: y_k+1 > R. Find max k s.t. y_k+1 > R => y_k >= R. $k_{min} = \text{bisect_left}(y, R)$.
            # For any k from 0 to $k_{min}-1$: y_k < R. Condition y_k+1 > R holds.
            # For k from 0 to $k_{min}-1$, and any j > k: Op 2 covers [y_k+1, y_j].
            k_min_op2_R = bisect_left(y, R)
            for k in range(k_min_op2_R):
                 for j in range(k + 1, m):
                      # Add edge (k, j) with cost 1, type 2, op i
                      adj[k].append((j, i, 2))

        # The number of edges is potentially O(m^2 * M). This is too many edges to build explicitly.

        # Let's optimize the DP transitions again.
        # $dp[j] = \min_{0 \le k < j, \text{edge}(k, j)} (dp[k] + 1)$.
        # For fixed j, calculate $dp[j]$.
        # $dp[j] = \min_{0 \le k < j} (dp[k] + 1)$ if edge $(k, j)$ exists.
        # Edge $(k, j)$ exists if $\exists i$: $(L_i \le y_k+1, R_i \ge y_j) \lor (y_j < L_i \lor y_k+1 > R_i)$.

        # $dp[j] = \min($
        #   $\min_{0 \le k < j} \{ dp[k] + 1 \mid \exists i: L_i \le y_k+1, R_i \ge y_j \}$, # Op 1 transition
        #   $\min_{0 \le k < j} \{ dp[k] + 1 \mid \exists i: y_j < L_i \}$,             # Op 2 transition (y_j < L_i)
        #   $\min_{0 \le k < j} \{ dp[k] + 1 \mid \exists i: y_k+1 > R_i \}$              # Op 2 transition (y_k+1 > R_i)
        # )

        # For fixed j:
        # Term 2: $\min \{ dp[k] + 1 \mid k < j, \exists i: y_j < L_i \}$. If there exists *any* op $i$ with $L_i > y_j$, this cost is $\min_{k<j} (dp[k] + 1)$. This feels wrong.
        # The cost should be 1 + min cost to cover [1, something else].

        # Let $dp[j]$ be min cost to cover $[1, y_j]$.
        # $dp[0] = 0$.
        # For $j=1 \dots m-1$:
        # $dp[j] = \infty$.
        # Option 1: Cover $y_j$ using Op 1 $i$ ($L_i \le y_j \le R_i$, $R_i \ge y_j$). Transition from $dp[\text{idx}(L_i-1)]$.
        # $dp[j] = \min_{i: L_i \le y_j, R_i \ge y_j} (dp[\text{idx}(L_i-1)] + 1)$. Store parent and op.
        # Option 2: Cover $y_j$ using Op 2 $i$ ($y_j < L_i$ or $y_j > R_i$).
        # If $y_j < L_i$: covers $[1, L_i-1]$. Transition from $dp[\text{idx}(L_i-1)]$? No. The whole range $[1, L_i-1]$ is covered.
        # This is tricky. The state must capture which parts are covered.

        # The sample output 2 0 1 0 implies operations can be chosen independently.
        # Total cost = $\sum_{i=1}^M \mathbb{I}(\text{op}_i 
e 0)$.
        # Final state is $\bigcup_{i: op_i=1} [L_i, R_i] \cup \bigcup_{i: op_i=2} ([1, L_i-1] \cup [R_i+1, N]) = [1, N]$.
        # Equivalently, $\bigcap_{i: op_i=2} [L_i, R_i] \cap [1, N] \subseteq \bigcup_{i: op_i=1} [L_i, R_i]$.

        # Iterate through all possible pairs of (L_S2, R_S2) achievable by $\bigcap_{i \in S_2} [L_i, R_i]$.
        # $L_{S_2} = \max_{i \in S_2} L_i$, $R_{S_2} = \min_{i \in S_2} R_i$.
        # Possible values for $L_{S_2}$ are $1$ or $L_i$. Possible values for $R_{S_2}$ are $N$ or $R_i$.
        # Let $U = \{1\} \cup \{L_i \mid 1 \le i \le M\}$. Let $V = \{N\} \cup \{R_i \mid 1 \le i \le M\}$.
        # For each $(\ell, r)$ with $\ell \in U, r \in V, 1 \le \ell \le r \le N$:
        # Check if $[\ell, r]$ can be formed as $\bigcap_{i \in S_2} [L_i, R_i]$ for some $S_2$.
        # This requires $S_2 \subseteq \{i \mid L_i \le \ell, R_i \ge r\}$.
        # And $\max_{i \in S_2} L_i = \ell$, $\min_{i \in S_2} R_i = r$.
        # Min cost to form $[\ell, r]$ as intersection:
        # cost = 1 if $\exists i: L_i=\ell, R_i=r$.
        # cost = 2 if $
eg (\exists i: L_i=\ell, R_i=r)$ AND $(\exists i_1: L_{i_1}=\ell, R_{i_1} \ge r)$ AND $(\exists i_2: L_{i_2} \le \ell, R_{i_2}=r)$.
        # cost = $\infty$ otherwise.

        # If cost $
e \infty$, need to cover $[\ell, r]$ with minimum number of type 1 operations from *all* $M$ operations.
        # Min cost to cover $[\ell, r]$ with Op 1s. Greedy approach $O(M \log M)$.
        # Total over $(|U| \times |V|)$ pairs: $O(M^2 \cdot M \log M)$. Too slow.

        # Let's try the DP on critical points $y_j$ representing prefix $[1, y_j]$.
        # $dp[j]$: min cost to cover $[1, y_j]$.
        # $dp[0]=0$. $dp[j]=\infty$ for $j>0$.
        # For $j=1 \ldots m-1$:
        # $dp[j] = dp[j-1] + 1$ # Using an Op 2 that covers y_j? This is not clear.

        # Final attempt at simple DP on critical points.
        # $dp[j]$: min cost to cover $[1, y_j]$.
        # $dp[0] = 0$.
        # For $j=1 \dots m-1$:
        # $dp[j] = dp[j-1]$. # Initialize based on covering [1, y_j-1]. This assumes y_j is somehow covered for free, which is incorrect.
        # $dp[j] = \min_{0 \le k < j} (dp[k] + \text{min_ops_to_cover_interval}(y_k+1, y_j))$
        # Let min_ops_to_cover_interval(a, b) be the minimum number of operations (type 1 or type 2) from the original set $\{1, \ldots, M\}$ whose union covers the interval $[a, b]$.
        # This is NOT correct. An operation can be type 1 or 2, and its effect is global.

        # Let's use the shortest path DP on critical points $y_0=0, y_1, \ldots, y_m=N+1$.
        # Node $j$ represents state where coverage extends up to $y_j$.
        # $dp[j]$: min cost to ensure all positions $1, \ldots, y_j$ are covered.
        # $dp[0] = 0$.
        # For $j = 1 \ldots m-1$:
        # $dp[j] = \infty$.
        # Iterate $k$ from $0$ to $j-1$.
        # If we have covered up to $y_k$ with cost $dp[k]$, we need to extend coverage to $y_j$.
        # The crucial requirement is that all points in $(y_k, y_j]$ must be covered.
        # Can we cover $(y_k, y_j]$ with one operation?
        # - Op 1 on $[L_i, R_i]$: if $L_i \le y_k+1$ and $R_i \ge y_j$. Cost 1.
        # - Op 2 on $[L_i, R_i]$: if $y_j < L_i$ or $y_k+1 > R_i$. Cost 1.
        # If such an $i$ exists, we can transition from $dp[k]$ to $dp[j]$ with cost 1.
        # $dp[j] = \min_{0 \le k < j, \exists i \text{ covers } [y_k+1, y_j]} (dp[k] + 1)$.

        # This formulation $dp[j]$ = shortest path from node 0 to node $j$ seems the most promising.
        # Node $j$ is index of critical point $y_j$. Target is node $m-1$ ($y_{m-1}=N+1$, effectively covers $[1, N]$).
        # $dp[0]=0$, $dp[j]=\infty$ for $j>0$.
        # For $j=1 \ldots m-1$:
        # For $k=0 \ldots j-1$:
        # Check if edge (k, j) exists: $\exists i$ s.t. ($L_i \le y[k]+1$ and $R_i \ge y[j]$) OR ($y[j] < L_i$ or $y[k]+1 > R_i$).
        # If edge exists and $dp[k] + 1 < dp[j]$: $dp[j] = dp[k] + 1$, parent[j] = k, parent_op_info[j] = (op_idx, op_type).

        # Optimization for edge check during DP:
        # For fixed $j$, iterate $k=j-1 \ldots 0$. $y_k+1$ decreases.
        # Needed for Op 1: $\max \{R_i \mid L_i \le y_k+1\}$. Use precomputed max_reach.
        # Needed for Op 2 (y_k+1 > R_i): $\max \{R_i \mid R_i < y_k+1\}$.
        # Needed for Op 2 (y_j < L_i): $\min \{L_i \mid L_i > y_j\}$. Constant for fixed $j$.

        # Precompute:
        # 1. max_R_op1[p] = max {R_i | L_i <= p}. And corresponding op index. $O(N+M)$.
        # 2. min_L_op2_gt[p] = min {L_i | L_i > p}. And corresponding op index. $O(N+M)$.
        # 3. max_R_op2_lt[p] = max {R_i | R_i < p}. And corresponding op index. $O(N+M)$.

        max_r_op1 = [-1] * (N + 2)
        op_idx_at_max_r_op1 = [-1] * (N + 2)
        current_max_r = -1
        current_op_idx = -1
        op_list_sorted_by_l = sorted([(L, R, i) for i, (L, R) in enumerate(ops)])
        op_idx = 0
        for p in range(1, N + 2):
             while op_idx < M and op_list_sorted_by_l[op_idx][0] <= p -1:
                 L, R, i = op_list_sorted_by_l[op_idx]
                 if R > current_max_r:
                     current_max_r = R
                     current_op_idx = i
                 op_idx += 1
             max_r_op1[p] = current_max_r
             op_idx_at_max_r_op1[p] = current_op_idx

        min_l_op2_gt = [N + 2] * (N + 2)
        op_idx_at_min_l_op2_gt = [-1] * (N + 2)
        current_min_l = N + 2
        current_op_idx = -1
        op_list_sorted_by_l_rev = sorted([(L, i) for i, (L, R) in enumerate(ops)], reverse=True)
        op_idx = 0
        for p in range(N + 1, -1, -1):
             while op_idx < M and op_list_sorted_by_l_rev[op_idx][0] > p:
                 L, i = op_list_sorted_by_l_rev[op_idx]
                 if L < current_min_l:
                     current_min_l = L
                     current_op_idx = i
                 op_idx += 1
             min_l_op2_gt[p] = current_min_l
             op_idx_at_min_l_op2_gt[p] = current_op_idx

        max_r_op2_lt = [-1] * (N + 2)
        op_idx_at_max_r_op2_lt = [-1] * (N + 2)
        current_max_r = -1
        current_op_idx = -1
        op_list_sorted_by_r = sorted([(R, i) for i, (L, R) in enumerate(ops)])
        op_idx = 0
        for p in range(1, N + 2):
             while op_idx < M and op_list_sorted_by_r[op_idx][0] < p - 1: # R < p-1
                 R, i = op_list_sorted_by_r[op_idx]
                 if R > current_max_r:
                     current_max_r = R
                     current_op_idx = i
                 op_idx += 1
             max_r_op2_lt[p] = current_max_r
             op_idx_at_max_r_op2_lt[p] = current_op_idx

        # DP calculation:
        dp = [float('inf')] * m
        parent = [None] * m
        parent_op_info = [None] * m # (op_idx, op_type)

        dp[0] = 0

        for j in range(1, m): # Iterate through target critical points y[j]
            # Check transitions from k < j
            # Optimization: Instead of iterating k, iterate through ops
            # An op i can create transitions to j.
            # Op 1 (L_i, R_i): Creates edge (k, j) if L_i <= y_k+1 and R_i >= y_j.
            #   Find min k such that L_i <= y_k+1 => y_k >= L_i-1. $k_{min} = \text{bisect_left}(y, L_i-1)$.
            #   Find max k such that L_i <= y_k+1 => y_k <= L_i-1. $k_{max} = \text{idx}(L_i-1)$.
            #   Find min j such that R_i >= y_j. $j_{min} = \text{bisect_left}(y, R_i)$.
            #   Op i can transition from k to j if $L_i \le y_k+1$ and $R_i \ge y_j$.
            #   This means $y_k \ge L_i-1$ and $y_j \le R_i$.
            #   Consider ops one by one to update DP?

            # DP using shortest path formulation with optimized transitions
            # $dp[j] = \min_{0 \le k < j} (dp[k] + 1 \text{ if edge } (k, j) \text{ exists})$
            # $dp[j] = \min ($
            #   $\min_{0 \le k < j \text{ s.t. } \exists i: L_i \le y_k+1, R_i \ge y_j} (dp[k]+1),$
            #   $\min_{0 \le k < j \text{ s.t. } \exists i: y_j < L_i \text{ (let this op be } i_0) } (dp[k]+1),$
            #   $\min_{0 \le k < j \text{ s.t. } \exists i: y_k+1 > R_i \text{ (let this op be } i_1) } (dp[k]+1)$
            # )

            # For fixed j, iterate k from j-1 down to 0.
            # Check Op 1 from k to j: $\exists i: L_i \le y_k+1, R_i \ge y_j$.
            # This is true if $\max \{R_i \mid L_i \le y_k+1 \} \ge y_j$.
            # max_r_op1[y_k+1] >= y[j].
            if y[k] + 1 <= N + 1 and y[j] >= 1 and max_r_op1[y[k]+1] >= y[j]:
                 if dp[k] + 1 < dp[j]:
                     dp[j] = dp[k] + 1
                     parent[j] = k
                     # Find *an* op i that satisfies L_i <= y_k+1 and R_i >= y_j
                     # The op stored in op_idx_at_max_r_op1[y[k]+1] has L <= y_k+1 and max R. If its R >= y_j, use it.
                     # This op gives the max R.
                     op_idx = op_idx_at_max_r_op1[y[k]+1]
                     if op_idx != -1 and ops[op_idx][1] >= y[j]:
                         parent_op_info[j] = (op_idx, 1)
                     else:
                         # Find *any* op i with L_i <= y_k+1 and R_i >= y_j. This is slow O(M).
                         # To make it O(1) on average, we can store representative ops.
                         # For reconstruction, any such op is fine. We need to find one.
                         # Store the one giving max_reach when precomputing.
                         # We already stored it in op_idx_at_max_r_op1
                         pass # op_idx_at_max_r_op1[y[k]+1] is already checked if >= y[j]


            # Check Op 2 from k to j: $\exists i: y_j < L_i$ or $y_k+1 > R_i$.
            # Condition $\exists i: y_j < L_i$ is constant for fixed j. True if min_l_op2_gt[y[j]] <= N.
            # Condition $\exists i: y_k+1 > R_i$ is true if max_r_op2_lt[y[k]+1] >= 1.
            op2_covers = False
            op2_idx = -1
            op2_type = -1

            # Case 1: y_j < L_i
            if min_l_op2_gt[y[j]] <= N: # Check if there exists an op with L > y_j
                 # Any such op i means Op 2 on i covers [1, L_i-1]. If L_i-1 >= y_j, covers [1, y_j].
                 # This corresponds to a transition from ANY k < j to j.
                 # The minimal cost is min(dp[k]) + 1 over k < j.
                 # This means dp[j] = min(dp[j], min(dp[0...j-1]) + 1).
                 # This minimum is just dp[j-1]? No, it could be dp[0].
                 # min_dp_prefix = min(dp[0...j-1]).
                 pass # Handle this outside the loop over k?

            # Case 2: y_k+1 > R_i
            if y[k]+1 >= 1 and max_r_op2_lt[y[k]+1] >= 0:
                 # There exists an op i with R_i < y_k+1. Op 2 on i covers [R_i+1, N].
                 # If R_i+1 <= y_k+1, this covers [y_k+1, N].
                 # This covers [y_k+1, y_j].
                 # This transition exists if max_r_op2_lt[y[k]+1] < y_k+1.
                 if max_r_op2_lt[y[k]+1] < y[k]+1: # Check if there is ANY R_i < y_k+1
                    op2_covers = True
                    op2_idx = op_idx_at_max_r_op2_lt[y[k]+1]
                    op2_type = 2

            # Op 2 covers [y_k+1, y_j] if $\exists i: y_j < L_i$ OR $\exists i: y_k+1 > R_i$.
            # This is NOT correct. One single op must cover the interval.
            # $\exists i: (y_j < L_i \text{ or } y_k+1 > R_i)$.

            op2_edge_exists = False
            op2_edge_idx = -1

            # Check Op 2: exists i s.t. y[j] < L_i or y[k]+1 > R_i
            # We need ONE such operation.
            # Check if min L_i > y[j]:
            op_idx1 = op_idx_at_min_l_op2_gt[y[j]]
            if op_idx1 != -1:
                # Found op i with L_i > y_j. This op satisfies y_j < L_i. Edge exists.
                op2_edge_exists = True
                op2_edge_idx = op_idx1

            # If not found yet, check if max R_i < y_k+1:
            if not op2_edge_exists:
                 op_idx2 = op_idx_at_max_r_op2_lt[y[k]+1]
                 if op_idx2 != -1:
                      # Found op i with R_i < y_k+1. This op satisfies y_k+1 > R_i. Edge exists.
                      op2_edge_exists = True
                      op2_edge_idx = op_idx2

            if op2_edge_exists:
                 if dp[k] + 1 < dp[j]:
                     dp[j] = dp[k] + 1
                     parent[j] = k
                     parent_op_info[j] = (op2_edge_idx, 2)


            # Need to handle the case where Op 1 covers [y_k+1, y_j] separately.
            # The Op 1 case was: max_r_op1[y[k]+1] >= y[j] AND corresponding op exists and has R >= y[j].
            # Let op_idx1 = op_idx_at_max_r_op1[y[k]+1]. If op_idx1 != -1 and ops[op_idx1][1] >= y[j]:
            # This op covers [y[k]+1, y[j]] with type 1.
            op_idx1 = op_idx_at_max_r_op1[y[k]+1]
            if op_idx1 != -1 and ops[op_idx1][1] >= y[j]:
                 if dp[k] + 1 < dp[j]:
                     dp[j] = dp[k] + 1
                     parent[j] = k
                     parent_op_info[j] = (op_idx1, 1)

        # End loop k
    # End loop j

    # Final answer is dp[m-1] (cost to cover [1, y[m-1]] = [1, N+1], effectively [1, N])
    if dp[m-1] == float('inf'):
        print(-1)
    else:
        print(dp[m-1])
        result_ops = [0] * M
        curr = m - 1
        while curr > 0:
            p_idx = parent[curr]
            op_idx, op_type = parent_op_info[curr]
            result_ops[op_idx] = op_type
            curr = p_idx

        print(*result_ops)


solve()