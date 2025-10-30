import sys

def main():
    input_data = sys.stdin.read().strip().split()
    T = int(input_data[0])
    
    # A quick function to count how many nodes are exactly distance d down
    # from a node x in this "parent = floor(i/2)" tree (1 <= node <= N).
    #
    # In an infinite full binary tree, all nodes at distance d below x
    # would lie in [x*2^d, x*2^d + 2^d - 1].
    # We intersect that range with [1, N] to account for boundaries.
    def count_dist(x, d, N):
        if d < 0:
            return 0
        # left = x << d, right = (x << d) + (2^d - 1)
        left = x << d
        right = left + ((1 << d) - 1)
        if left > N:
            return 0
        L = max(left, 1)
        R = min(right, N)
        if R < L:
            return 0
        return R - L + 1
    
    # Process each test
    idx = 1
    out = []
    for _ in range(T):
        N = int(input_data[idx]);   idx += 1
        X = int(input_data[idx]);   idx += 1
        K = int(input_data[idx]);   idx += 1
        
        # Collect ancestors of X (including X itself), up to the root (1)
        ancestors = []
        cur = X
        while cur > 0:
            ancestors.append(cur)
            cur //= 2
        # Example: if X=10, ancestors = [10, 5, 2, 1]
        
        h = len(ancestors) - 1  # h is the last index => distance from X to the root
        
        answer = 0
        # For each j = 0..min(K,h), consider going up j steps, then down K-j steps
        # LCA = ancestors[j], subtract the subtree going down from ancestors[j-1] if j>0
        limit = min(K, h)
        for j in range(limit + 1):
            # Add nodes that are distance (K-j) down from A_j
            A_j = ancestors[j]
            answer += count_dist(A_j, K - j, N)
            # Subtract nodes that lie in the subtree of the child leading to X
            if j > 0:
                # That child is ancestors[j-1], distance from A_j is 1 + ...
                answer -= count_dist(ancestors[j - 1], (K - j) - 1, N)
        
        out.append(str(answer))
    
    print("
".join(out))

# Don't forget to call main!
main()