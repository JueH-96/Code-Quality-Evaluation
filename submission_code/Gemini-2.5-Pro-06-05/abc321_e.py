import sys

def solve():
    """
    Solves a single test case for the tree distance problem.
    """
    try:
        line = sys.stdin.readline()
        if not line:
            return
        N, X, K = map(int, line.split())
    except (IOError, ValueError):
        return

    # The problem defines a binary tree where the parent of vertex i is floor(i/2).
    # This means vertex 1 is the root.
    # The path between two vertices u and v passes through their Lowest Common Ancestor (LCA).
    # We can categorize vertices 'v' based on their LCA with 'X'.

    # Case 0: K = 0
    # The only vertex at distance 0 from X is X itself.
    if K == 0:
        print(1)
        return

    ans = 0

    # Case 1: LCA(v, X) = X
    # This means v is a descendant of X. dist(v, X) = K.
    # Descendants of a node `p` at distance `d` are in the range [p*2^d, (p+1)*2^d - 1].
    # We count how many are within the valid vertex range [1, N].
    low = X << K
    if low <= N:
        high = (X + 1) << K
        # Count vertices in [low, high-1] that are also <= N
        count = min(N, high - 1) - low + 1
        ans += count

    # Case 2: LCA(v, X) is a proper ancestor of X.
    # We iterate up from X. `child_on_path` is the node on the path from X to the root.
    child_on_path = X
    dist_X_to_child = 0

    while child_on_path > 1 and dist_X_to_child < K:
        parent_of_child = child_on_path // 2
        
        # For `parent_of_child` to be the LCA, v must be a descendant of `sibling`.
        sibling = child_on_path ^ 1

        # The distance from X to this LCA is `dist_X_to_child + 1`.
        dist_X_to_LCA = dist_X_to_child + 1

        # The remaining distance to go "down" from the LCA to v.
        rem_dist_down = K - dist_X_to_LCA
        
        # The path from v to the LCA goes through the sibling.
        # dist(v, LCA) = dist(v, sibling) + 1.
        # So, rem_dist_down = dist(v, sibling) + 1.
        dist_from_sibling = rem_dist_down - 1
        
        if dist_from_sibling < 0:
            # This happens when rem_dist_down = 0, which means dist(X, LCA) = K.
            # The node v is the LCA itself (`parent_of_child`).
            # It's a valid vertex, so we add 1 to the count.
            ans += 1
        else:
            # v is a descendant of `sibling` at distance `dist_from_sibling`.
            # We count how many such descendants are <= N.
            low = sibling << dist_from_sibling
            if low <= N:
                high = (sibling + 1) << dist_from_sibling
                count = min(N, high - 1) - low + 1
                ans += count
        
        # Move one step up the path from X.
        dist_X_to_child += 1
        child_on_path = parent_of_child
    
    print(ans)


def main():
    try:
        T_str = sys.stdin.readline()
        if not T_str:
            return
        T = int(T_str)
        for _ in range(T):
            solve()
    except (IOError, ValueError):
        # Gracefully handle empty or malformed input
        pass

if __name__ == "__main__":
    main()