import sys

# It's good practice to set recursion limit for segment tree,
# although for N=2e5, logN is small enough that default limit (1000-3000) is fine.
# Example: sys.setrecursionlimit(10**6) 

# Segment tree functions
# A_list: input array of gourmet levels (0-indexed)
# seg_tree: array to store segment tree nodes
# node_idx: current node index in seg_tree (1-based for convenience)
# range_L, range_R: segment tree node covers indices [range_L, range_R] of A_list

def build_seg_tree(A_list, seg_tree, node_idx, range_L, range_R):
    if range_L == range_R:
        seg_tree[node_idx] = A_list[range_L]
    else:
        mid = (range_L + range_R) // 2
        build_seg_tree(A_list, seg_tree, 2 * node_idx, range_L, mid)
        build_seg_tree(A_list, seg_tree, 2 * node_idx + 1, mid + 1, range_R)
        seg_tree[node_idx] = min(seg_tree[2 * node_idx], seg_tree[2 * node_idx + 1])

# Query function to find the smallest index 'i' such that A_list[i] <= B_val
# Returns 0-indexed person index, or -1 if no such person exists.
def query_seg_tree(seg_tree, node_idx, range_L, range_R, B_val):
    # If min gourmet level in this range is already greater than B_val,
    # no one in this range can eat the sushi.
    if seg_tree[node_idx] > B_val:
        return -1
    
    # If this is a leaf node
    if range_L == range_R:
        # We know A_list[range_L] (which is seg_tree[node_idx]) <= B_val
        # from the check above (or it's implied if not seg_tree[node_idx] > B_val).
        # So, this person can eat the sushi.
        return range_L # This is the 0-indexed person index
    
    mid = (range_L + range_R) // 2
    
    # Try left child first. Its range is [range_L, mid].
    # The node index for left child is 2 * node_idx.
    res_left = query_seg_tree(seg_tree, 2 * node_idx, range_L, mid, B_val)
    if res_left != -1:
        return res_left # Found in left subtree, this is the first one
    
    # If not found in left child, try right child. Its range is [mid + 1, range_R].
    # The node index for right child is 2 * node_idx + 1.
    res_right = query_seg_tree(seg_tree, 2 * node_idx + 1, mid + 1, range_R, B_val)
    return res_right # Result from right subtree (could be -1)

def main():
    # Fast I/O
    input_func = sys.stdin.readline

    N, M = map(int, input_func().split())
    A_list = list(map(int, input_func().split())) # Gourmet levels of N people
    B_list = list(map(int, input_func().split())) # Deliciousness of M sushis

    # Initialize segment tree array
    # Size: 4*N is a safe upper bound for segment tree array.
    # Values can be anything initially, build_seg_tree will populate them.
    # Max N = 2*10^5, so 4*N = 8*10^5.
    seg_tree = [0] * (4 * N) 
    
    # Constraints: 1 <= N, M. So N is at least 1.
    # A_list will have N elements, indices 0 to N-1.
    build_seg_tree(A_list, seg_tree, 1, 0, N - 1)

    results_str_list = []
    for B_val in B_list:
        found_idx_0_based = query_seg_tree(seg_tree, 1, 0, N - 1, B_val)
        
        if found_idx_0_based == -1:
            results_str_list.append("-1")
        else:
            # Convert to 1-indexed person number and then to string
            results_str_list.append(str(found_idx_0_based + 1)) 
            
    sys.stdout.write("
".join(results_str_list) + "
")

if __name__ == '__main__':
    main()