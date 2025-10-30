import sys

# It's good practice to increase recursion limit for segment tree or other recursive structures if N is large.
# sys.setrecursionlimit(10**6) # Default is often 1000 or 3000. Max N implies logN depth. Fine.

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    A_list = list(map(int, sys.stdin.readline().split()))
    B_list = list(map(int, sys.stdin.readline().split()))

    dishes = []
    for i in range(N):
        dishes.append({'a': A_list[i], 'b': B_list[i], 'id': i})

    dishes_sorted_A = sorted(dishes, key=lambda d: d['a'], reverse=True)
    
    # b_dishes: dishes sorted by B (ascending), then A (ascending for tie-breaking)
    # This order is used by the segment tree. Leaf j of segtree corresponds to b_dishes[j].
    b_dishes = sorted(dishes, key=lambda d: (d['b'], d['a']))
    dish_to_b_idx = {d['id']: i for i, d in enumerate(b_dishes)}

    # Segment tree implementation
    # Tree stores tuples: (count_active_dishes, sum_A_active, sum_B_active)
    # It's 1-indexed array for easier parent/child calculation (i -> 2i, 2i+1)
    # Leaves are at indices N to 2N-1. Root is at index 1.
    st_val = [(0,0,0)] * (2 * N)

    def st_combine(res1, res2):
        return (res1[0] + res2[0], res1[1] + res2[1], res1[2] + res2[2])

    def st_build():
        for i in range(N):
            st_val[N + i] = (1, b_dishes[i]['a'], b_dishes[i]['b'])
        for i in range(N - 1, 0, -1):
            st_val[i] = st_combine(st_val[i * 2], st_val[i * 2 + 1])

    def st_modify(idx_in_b_dishes, active):
        # idx_in_b_dishes is 0 to N-1
        p = idx_in_b_dishes + N # actual position in st_val array
        if active: # This case not used in current check(), but good for general segtree
            st_val[p] = (1, b_dishes[idx_in_b_dishes]['a'], b_dishes[idx_in_b_dishes]['b'])
        else: # Deactivate
            st_val[p] = (0, 0, 0)
        
        while p > 1:
            st_val[p // 2] = st_combine(st_val[p], st_val[p ^ 1]) # p^1 is sibling
            p //= 2
    
    # Query for m_items active dishes with largest B values.
    # u is node index in st_val (1-indexed), k_needed is number of items.
    def st_query_k_largest_b_recursive(u, k_needed):
        if k_needed <= 0:
            return (0, 0, 0)
        
        if u >= N: # Leaf node (indices N to 2N-1 are leaves)
            # If leaf is active (count=1) and we need items, take it.
            # (A leaf contributes at most 1 item to k_needed)
            if st_val[u][0] > 0: # Active
                return st_val[u]
            else: # Not active
                return (0, 0, 0)

        # Internal node u. Children are 2*u (left) and 2*u+1 (right).
        # Try right child first (corresponds to larger B values in b_dishes).
        count_in_right, sum_a_in_right, sum_b_in_right = st_val[2 * u + 1]

        if count_in_right >= k_needed:
            # All k_needed items can be found in right subtree
            return st_query_k_largest_b_recursive(2 * u + 1, k_needed)
        else:
            # Take all 'count_in_right' items from right subtree.
            # Their total contribution is (count_in_right, sum_a_in_right, sum_b_in_right).
            # Need 'k_needed - count_in_right' more items from left subtree.
            res_left = st_query_k_largest_b_recursive(2 * u, k_needed - count_in_right)
            return st_combine((count_in_right, sum_a_in_right, sum_b_in_right), res_left)

    def check(k_target_dishes):
        if k_target_dishes == 0: return False
        if k_target_dishes > N: return False # Cannot pick more dishes than available

        st_build() # Initialize segment tree, all dishes active

        current_A_sum_chosen = 0 # Sums for s_A dishes chosen from dishes_sorted_A
        current_B_sum_chosen = 0
        
        for s_A in range(k_target_dishes + 1): # s_A is count of dishes taken for A-value
            dishes_to_pick_by_B = k_target_dishes - s_A
            
            # Query for `dishes_to_pick_by_B` items from segment tree (active dishes)
            count_seg, sum_A_seg, sum_B_seg = (0,0,0)
            if dishes_to_pick_by_B > 0 :
                 count_seg, sum_A_seg, sum_B_seg = st_query_k_largest_b_recursive(1, dishes_to_pick_by_B)

            # Check if we successfully found k_target_dishes in total
            if s_A + count_seg >= k_target_dishes:
                total_A = current_A_sum_chosen + sum_A_seg
                total_B = current_B_sum_chosen + sum_B_seg
                if total_A > X or total_B > Y:
                    return True
            
            # Prepare for next iteration (s_A+1 dishes chosen for A-value)
            if s_A < k_target_dishes:
                if s_A >= N: # No more dishes left in dishes_sorted_A
                    break 
                
                # The (s_A)-th dish (0-indexed) from dishes_sorted_A is now chosen.
                dish_newly_chosen_for_A = dishes_sorted_A[s_A]
                current_A_sum_chosen += dish_newly_chosen_for_A['a']
                current_B_sum_chosen += dish_newly_chosen_for_A['b']
                
                # This dish is no longer available for B-selection, so deactivate in segtree.
                idx_in_b_sorted_list = dish_to_b_idx[dish_newly_chosen_for_A['id']]
                st_modify(idx_in_b_sorted_list, False) # False means deactivate
        
        return False

    # Binary search for the minimum k
    low = 1
    high = N
    ans = N + 1 # Initialize with a value larger than N

    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
            
    print(ans)

solve()