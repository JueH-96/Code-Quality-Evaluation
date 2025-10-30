import sys
import heapq # Not directly used for main logic, but good to remember for heaps
import bisect # For bisect_left

# It's good practice to set higher recursion limit for segment tree
sys.setrecursionlimit(4 * 10**5) 

def solve():
    N, M = map(int, sys.stdin.readline().split())
    P_prices = list(map(int, sys.stdin.readline().split()))
    L_thresholds = list(map(int, sys.stdin.readline().split()))
    D_discounts = list(map(int, sys.stdin.readline().split()))

    coupons = []
    for i in range(M):
        coupons.append((D_discounts[i], L_thresholds[i])) # (D, L) format

    # Sort coupons: D desc, then L asc
    coupons.sort(key=lambda x: (-x[0], x[1]))

    # Sort item prices for segment tree base
    # Store as (price, original_index_in_P_prices) if needed, but not here.
    # Just prices, as they are indistinguishable beyond value.
    # However, if multiple items have same price, they are distinct items.
    # So, we need to sort P_prices and build segment tree over these N items.
    # S_P will be the sorted list of N item prices.
    S_P = sorted(P_prices)

    # Segment Tree stores (min_price_in_range, index_in_S_P_of_that_min_price)
    # unavailable items are marked with (float('inf'), -1)
    # Size of S_P is N.
    
    if N == 0: # No items, cost is 0
        print(0)
        return

    seg_tree_data = [(float('inf'), -1)] * (4 * N)

    def build_seg_tree(node_idx, current_range_low, current_range_high):
        if current_range_low == current_range_high:
            # Leaf node corresponds to item S_P[current_range_low]
            seg_tree_data[node_idx] = (S_P[current_range_low], current_range_low)
        else:
            mid = (current_range_low + current_range_high) // 2
            build_seg_tree(2 * node_idx, current_range_low, mid)
            build_seg_tree(2 * node_idx + 1, mid + 1, current_range_high)
            seg_tree_data[node_idx] = min(seg_tree_data[2 * node_idx], seg_tree_data[2 * node_idx + 1])

    # Query for minimum (price, index) in S_P within query_range [ql, qh]
    def query_seg_tree(node_idx, current_range_low, current_range_high, ql, qh):
        if ql > qh: # Empty query range
            return (float('inf'), -1)
        if ql <= current_range_low and current_range_high <= qh: # Current segment fully within query range
            return seg_tree_data[node_idx]
        
        mid = (current_range_low + current_range_high) // 2
        res_left = (float('inf'), -1)
        if ql <= mid: # Overlap with left child
            res_left = query_seg_tree(2 * node_idx, current_range_low, mid, ql, qh)
        
        res_right = (float('inf'), -1)
        if qh > mid: # Overlap with right child
            res_right = query_seg_tree(2 * node_idx + 1, mid + 1, current_range_high, ql, qh)
            
        return min(res_left, res_right)

    # Update segment tree: mark item at S_P_idx as unavailable
    def update_seg_tree(node_idx, current_range_low, current_range_high, S_P_idx_to_update):
        if current_range_low == current_range_high: # Leaf node
            seg_tree_data[node_idx] = (float('inf'), -1) # Mark as unavailable
        else:
            mid = (current_range_low + current_range_high) // 2
            if S_P_idx_to_update <= mid:
                update_seg_tree(2 * node_idx, current_range_low, mid, S_P_idx_to_update)
            else:
                update_seg_tree(2 * node_idx + 1, mid + 1, current_range_high, S_P_idx_to_update)
            seg_tree_data[node_idx] = min(seg_tree_data[2 * node_idx], seg_tree_data[2 * node_idx + 1])

    build_seg_tree(1, 0, N - 1)

    total_discount = 0
    for D_val, L_val in coupons:
        # Find smallest index idx_L in S_P such that S_P[idx_L] >= L_val
        idx_L = bisect.bisect_left(S_P, L_val)
        
        if idx_L == N: # All remaining items are cheaper than L_val
            continue

        # Query segment tree for an available item P_k >= L_val
        # This means querying in range [idx_L, N-1] of S_P indices
        chosen_item_price, chosen_item_S_P_idx = query_seg_tree(1, 0, N - 1, idx_L, N - 1)
        
        if chosen_item_price != float('inf'): # If an applicable item is found
            total_discount += D_val
            update_seg_tree(1, 0, N - 1, chosen_item_S_P_idx) # Mark item as used
            
    initial_total_cost = sum(P_prices)
    final_total_cost = initial_total_cost - total_discount
    print(final_total_cost)

solve()