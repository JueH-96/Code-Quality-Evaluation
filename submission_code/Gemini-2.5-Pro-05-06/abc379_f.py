import sys

# It's good practice to set a higher recursion limit for segment tree operations
# if N can be large, though Python's default (1000-3000) is often sufficient
# for O(log N) depth recursion like in segment trees.
# Max depth for N=2e5 is ~18-20 for basic ops.
# Setting to a moderate value; extremely high values can be problematic.
sys.setrecursionlimit(5000) 

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    # Using 1-based indexing for H to match problem statement
    H = [0] + list(map(int, sys.stdin.readline().split())) 

    queries_by_L = [[] for _ in range(N + 1)]
    for i in range(Q):
        l, r = map(int, sys.stdin.readline().split())
        queries_by_L[l].append((r, i))

    # Calculate Next Greater Element (NGE)
    # nge[i] = smallest k > i such that H[k] > H[i].
    # If no such k, nge[i] = N + 1.
    nge = [N + 1] * (N + 2) 
    stack = [] 
    for i in range(1, N + 1):
        while stack and H[stack[-1]] < H[i]: 
            idx = stack.pop()
            nge[idx] = i 
        stack.append(i)
    
    # Segment Tree setup
    tree_size = 4 * (N + 1) 
    seg_tree_sum = [0] * tree_size
    seg_tree_lazy = [-1] * tree_size # -1: no lazy, 0: set to 0, 1: set to 1

    def push(v_node, tl, tr):
        if seg_tree_lazy[v_node] != -1 and tl != tr: 
            val_to_set = seg_tree_lazy[v_node]
            tm = (tl + tr) // 2
            
            seg_tree_lazy[2 * v_node] = val_to_set
            seg_tree_sum[2 * v_node] = val_to_set * (tm - tl + 1)
            
            seg_tree_lazy[2 * v_node + 1] = val_to_set
            seg_tree_sum[2 * v_node + 1] = val_to_set * (tr - (tm + 1) + 1)
            
            seg_tree_lazy[v_node] = -1

    def update_range_in_seg_tree(v_node, tl, tr, l, r, val_to_set):
        if l > r: 
            return
        
        push(v_node, tl, tr) 
        
        if tl > r or tr < l: 
            return

        if l <= tl and tr <= r: 
            seg_tree_sum[v_node] = val_to_set * (tr - tl + 1)
            if tl != tr: 
                seg_tree_lazy[v_node] = val_to_set
            return

        tm = (tl + tr) // 2
        update_range_in_seg_tree(2 * v_node, tl, tm, l, r, val_to_set)
        update_range_in_seg_tree(2 * v_node + 1, tm + 1, tr, l, r, val_to_set)
        
        seg_tree_sum[v_node] = seg_tree_sum[2 * v_node] + seg_tree_sum[2 * v_node + 1]

    def query_sum_from_seg_tree(v_node, tl, tr, l, r):
        if l > r: 
            return 0
        
        push(v_node, tl, tr)
        
        if tl > r or tr < l: 
            return 0
        
        if l <= tl and tr <= r: 
            return seg_tree_sum[v_node]

        tm = (tl + tr) // 2
        left_sum = query_sum_from_seg_tree(2 * v_node, tl, tm, l, r)
        right_sum = query_sum_from_seg_tree(2 * v_node + 1, tm + 1, tr, l, r)
        return left_sum + right_sum

    ans = [0] * Q

    for L_val in range(N, 0, -1):
        if L_val + 1 <= N:
            # Point update: building L_val+1 is visible from L_val. Set to 1.
            update_range_in_seg_tree(1, 1, N, L_val + 1, L_val + 1, 1)

        # Range update: buildings H[k] for L_val < k < nge[L_val] are shorter than H[L_val].
        # H[L_val] blocks view to them. Set to 0.
        # Range of indices is [L_val + 1, nge[L_val] - 1].
        if L_val + 1 <= nge[L_val] - 1: # Ensure range is valid before updating
            update_range_in_seg_tree(1, 1, N, L_val + 1, nge[L_val] - 1, 0)
        
        if queries_by_L[L_val]:
            for R_val, query_idx in queries_by_L[L_val]:
                # Count buildings k such that R_val < k <= N. Query sum in [R_val + 1, N].
                count = query_sum_from_seg_tree(1, 1, N, R_val + 1, N)
                ans[query_idx] = count
    
    sys.stdout.write('
'.join(map(str, ans)) + '
')

solve()