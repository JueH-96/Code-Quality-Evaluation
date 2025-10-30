import sys

sys.setrecursionlimit(4 * 10**5) # Increased recursion limit

def main():
    N, Q = map(int, sys.stdin.readline().split())

    counts = [0] * (N + 1)
    
    st_n = 1
    while st_n < N:
        st_n *= 2
    
    tree = [0] * (2 * st_n) # tree[u]: color if monochromatic and no lazy tag, else 0 for mixed
    lazy = [0] * (2 * st_n) # lazy[u]: pending update color for range u

    def build(u, s, e):
        if s == e:
            if s <= N:
                tree[u] = s 
                counts[s] = 1
            return
        
        mid = (s + e) // 2
        build(2 * u, s, mid)
        build(2 * u + 1, mid + 1, e)
        tree[u] = 0 # Mixed (e.g. cell 1 color 1, cell 2 color 2)

    build(1, 1, st_n)

    def push(u):
        if lazy[u] != 0:
            # Children become monochromatic with lazy[u]'s color
            tree[2*u] = lazy[u]
            lazy[2*u] = lazy[u]
            tree[2*u+1] = lazy[u]
            lazy[2*u+1] = lazy[u]
            lazy[u] = 0

    def query_color_at_idx(u, s, e, idx):
        if lazy[u] != 0: # Overriding lazy tag
            return lazy[u]
        if s == e: # Leaf node
            return tree[u]

        mid = (s + e) // 2
        # No push(u) here because if lazy[u] was set, we'd have returned.
        # If lazy[u] is 0, tree[u] reflects children state or is leaf.
        if idx <= mid:
            return query_color_at_idx(2*u, s, mid, idx)
        else:
            return query_color_at_idx(2*u+1, mid+1, e, idx)

    def get_effective_color(u_node):
        # Gets the actual color of the range u_node considering its lazy tag
        return lazy[u_node] if lazy[u_node] != 0 else tree[u_node]

    def find_rightmost_not_color(u, s, e, search_s, search_e, target_color):
        # Base cases for search range
        if s > search_e or e < search_s or search_s > search_e:
            return 0 
        
        # Effective color of current segment [s,e]
        eff_color_u = get_effective_color(u)

        if eff_color_u != 0: # Current segment [s,e] is effectively monochromatic
            if eff_color_u != target_color:
                return min(e, search_e) 
            else: # eff_color_u == target_color
                return 0 
        
        # Current segment is mixed (eff_color_u == 0)
        if s == e: # Leaf node that is mixed - should not happen with current logic where 0 means mixed
             return 0 # All leaves should have a color or be overridden by lazy tag

        mid = (s + e) // 2
        res_k = find_rightmost_not_color(2*u+1, mid+1, e, search_s, search_e, target_color)
        if res_k != 0:
            return res_k
        res_k = find_rightmost_not_color(2*u, s, mid, search_s, search_e, target_color)
        return res_k

    def find_leftmost_not_color(u, s, e, search_s, search_e, target_color):
        if s > search_e or e < search_s or search_s > search_e:
            return N + 1
        
        eff_color_u = get_effective_color(u)

        if eff_color_u != 0: # Current segment [s,e] is effectively monochromatic
            if eff_color_u != target_color:
                return max(s, search_s)
            else: # eff_color_u == target_color
                return N + 1
        
        if s == e:
            return N + 1

        mid = (s + e) // 2
        res_k = find_leftmost_not_color(2*u, s, mid, search_s, search_e, target_color)
        if res_k != N + 1:
            return res_k
        res_k = find_leftmost_not_color(2*u+1, mid+1, e, search_s, search_e, target_color)
        return res_k
    
    def update_range_color(u, s, e, update_s, update_e, new_color):
        if s > update_e or e < update_s or update_s > update_e:
            return

        if update_s <= s and e <= update_e: 
            tree[u] = new_color
            lazy[u] = new_color
            return
        
        push(u) 

        mid = (s + e) // 2
        update_range_color(2*u, s, mid, update_s, update_e, new_color)
        update_range_color(2*u+1, mid+1, e, update_s, update_e, new_color)
        
        color_left = get_effective_color(2*u)
        color_right = get_effective_color(2*u+1)

        if color_left == color_right and color_left != 0 : # color_left != 0 ensures it's not "mixed"
            tree[u] = color_left
        else:
            tree[u] = 0 # Mixed

    results = []
    for i_query in range(Q):
        query_line = sys.stdin.readline().split()
        q_type = int(query_line[0])

        if q_type == 1:
            x, c_new = int(query_line[1]), int(query_line[2])
            
            c_old = query_color_at_idx(1, 1, st_n, x)

            if c_old == c_new:
                continue

            k_L = find_rightmost_not_color(1, 1, st_n, 1, x - 1, c_old)
            L = k_L + 1
            
            k_R = find_leftmost_not_color(1, 1, st_n, x + 1, N, c_old)
            R = k_R - 1
            
            num_changed = R - L + 1
            if num_changed > 0 : # Should always be true as x is in [L,R]
                counts[c_old] -= num_changed
                counts[c_new] += num_changed
                update_range_color(1, 1, st_n, L, R, c_new)

        else: # q_type == 2
            c = int(query_line[1])
            results.append(str(counts[c]))
            
    sys.stdout.write("
".join(results))
    # Add a final newline if problem expects it (usually yes for multi-line output)
    if results: # Avoid printing just a newline if no type 2 queries
        sys.stdout.write("
")

main()