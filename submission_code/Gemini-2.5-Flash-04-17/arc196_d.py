import sys

# Increase recursion depth for DSU find operation if needed
# In competitive programming platforms, default recursion depth might be small.
# sys.setrecursionlimit(2000) # This might not be necessary for this problem structure with path compression

class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))
        self.size = [1] * (n + 1)
        self.min_val = list(range(n + 1))
        self.max_val = list(range(n + 1))
        # Store lists of person indices (0 to M-1) whose u_i is in this set
        self.people_in_set = [[] for _ in range(n + 1)]

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i == root_j:
            return [], [] # No merge, no new failures from this union

        # Ensure root_i is the root of the larger set for merging people lists later
        if self.size[root_i] < self.size[root_j]:
            root_i, root_j = root_j, root_i # root_i is now the root of the larger set

        # Store old values and people lists BEFORE merging
        min_i_before = self.min_val[root_i]
        max_i_before = self.max_val[root_i]
        people_from_i = self.people_in_set[root_i]

        min_j_before = self.min_val[root_j]
        max_j_before = self.max_val[root_j]
        people_from_j = self.people_in_set[root_j]
        
        newly_failed = []

        # Check people in the smaller set (root_j) against the larger set (root_i)
        for person_idx in people_from_j:
            # We don't check here if person already failed. The caller does that.
            u_p, v_p = people[person_idx]
            # Person person_idx with range (u_p, v_p). u_p is currently in set root_j.
            # Failure if Set(root_i) has element in (u_p, v_p).
            # This is true iff min(Set(root_i)) < v_p AND max(Set(root_i)) > u_p
            if min_i_before < v_p and max_i_before > u_p:
                newly_failed.append(person_idx)

        # Check people in the larger set (root_i) against the smaller set (root_j)
        for person_idx in people_from_i:
            # We don't check here if person already failed. The caller does that.
            u_p, v_p = people[person_idx]
            # Person person_idx with range (u_p, v_p). u_p is currently in set root_i.
            # Failure if Set(root_j) has element in (u_p, v_p).
            # This is true iff min(Set(root_j)) < v_p AND max(Set(root_j)) > u_p
            if min_j_before < v_p and max_j_before > u_p:
                 newly_failed.append(person_idx)
        
        # Perform the actual merge (root_j into root_i, root_i is the larger set root)
        self.parent[root_j] = root_i
        self.size[root_i] += self.size[root_j]
        self.min_val[root_i] = min(self.min_val[root_i], self.min_val[root_j])
        self.max_val[root_i] = max(self.max_val[root_i], self.max_val[root_j])
        
        # Merge people lists: smaller list (people_from_j) into larger list (people_from_i)
        self.people_in_set[root_i].extend(people_from_j)
        self.people_in_set[root_j] = []

        return newly_failed

# Segment Tree for Range Minimum Query
class SegTree:
    def __init__(self, n, default_value):
        self.n = n
        # Tree size 4*n is standard for 1-based indexing or up to 2*n for 0-based power of 2
        # Using 4*n for safety with 1-based indexing on person indices 1..M
        self.tree = [default_value] * (4 * n + 4) 
        self.default_value = default_value

    def update(self, idx, val, node, l, r):
        if l == r:
            self.tree[node] = val
            return
        mid = (l + r) // 2
        if l <= idx <= mid:
            self.update(idx, val, 2 * node, l, mid)
        else:
            self.update(idx, val, 2 * node + 1, mid + 1, r)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, ql, qr, node, l, r):
        if qr < l or r < ql:
            return self.default_value
        if ql <= l and r <= qr:
            return self.tree[node]
        mid = (l + r) // 2
        left_min = self.query(ql, qr, 2 * node, l, mid)
        right_min = self.query(ql, qr, 2 * node + 1, mid + 1, r)
        return min(left_min, right_min)

# Read input
N, M, Q = map(int, sys.stdin.readline().split())
people_raw = []
for _ in range(M):
    S, T = map(int, sys.stdin.readline().split())
    people_raw.append((S, T))

# Store people as (min(S,T), max(S,T)) pairs
people = [(min(s, t), max(s, t)) for s, t in people_raw]

queries = []
for i in range(Q):
    L, R = map(int, sys.stdin.readline().split())
    queries.append((L, R, i))

# Sort queries by R
queries.sort(key=lambda x: x[1])

# Store queries grouped by R
queries_by_R = [[] for _ in range(M + 1)]
for L, R, q_idx in queries:
    queries_by_R[R].append((L, q_idx))

# Initialize DSU
dsu = DSU(N)

# Initialize Fail array and Segment Tree
# fail[i] stores the minimum R (1-indexed) where person index i (0-indexed) fails
# Initialized to M + 1, indicating no failure up to M
fail = [M + 1] * M 
seg_tree = SegTree(M, M + 1) # Segment tree handles indices 1 to M

# Populate initial people_in_set lists in DSU
# For each node j, people_by_node[j] stores indices i where u_i = j
people_by_node = [[] for _ in range(N + 1)]
for i in range(M):
    u, v = people[i]
    people_by_node[u].append(i)

# Assign people to initial sets (each node j is initially in set {j})
# DSU is 1-indexed for nodes 1..N
for j in range(1, N + 1):
    dsu.people_in_set[j] = people_by_node[j]

# Process people 1 to M
answers = [None] * Q

for R in range(1, M + 1):
    u_R, v_R = people[R - 1] # Person R is index R-1 (0-indexed)
    
    # Perform union and get list of people indices that might have failed
    # The DSU union logic now correctly handles checking people from both merging sets
    newly_failed_potential = dsu.union(u_R, v_R)

    # Update Fail values and Segment Tree for potentially failed people
    for person_idx in newly_failed_potential:
         if fail[person_idx] > M: # If this person hasn't failed yet at a step <= R
             fail[person_idx] = R
             # Update seg tree at index (person_idx + 1) with failure step R
             # Segment tree is 1-indexed on person indices (1 to M)
             seg_tree.update(person_idx + 1, R, 1, 1, M) 

    # Process queries with R_k = R
    for L, q_idx in queries_by_R[R]:
        # Query range [L, R] on people indices (1-based in problem)
        # Corresponds to indices L-1 to R-1 in the 0-indexed people list.
        # Segment tree indices are 1-based, corresponding to person indices L to R.
        min_fail_in_range = seg_tree.query(L, R, 1, 1, M)

        # If the minimum failure step in the query range [L, R]
        # is less than or equal to the current step R,
        # then at least one person in the range failed by step R.
        if min_fail_in_range <= R:
            answers[q_idx] = "No"
        else:
            answers[q_idx] = "Yes"

# Print answers
for ans in answers:
    print(ans)