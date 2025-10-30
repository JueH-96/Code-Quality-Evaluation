import sys

def main():
    """
    This function contains the full logic for the problem.
    It reads input, performs preprocessing, and answers queries.
    """
    # Increase recursion limit for the segment tree on large inputs.
    # While the recursion depth is O(log N), it's a good practice for safety.
    sys.setrecursionlimit(2 * 10**5 + 5)
    
    # Use fast I/O
    input = sys.stdin.readline

    try:
        N_str = input()
        if not N_str: return
        N = int(N_str)
        A = list(map(int, input().split()))
        Q = int(input())
    except (IOError, ValueError):
        return

    # --- Preprocessing ---

    # 1. Compute match_idx[j]: smallest index p > j where 2*A[j] <= A[p].
    # This is done in O(N) using a two-pointer approach since A is sorted.
    match_idx = [N] * N
    p = 1
    for j in range(N):
        p = max(p, j + 1)
        while p < N and 2 * A[j] > A[p]:
            p += 1
        if p < N:
            match_idx[j] = p
    
    # 2. Compute D[j] = match_idx[j] - j.
    # D[j] is the minimum distance an element must be "shifted" in the sorted
    # array to find a valid partner for A[j].
    # If no match exists (match_idx[j] == N), D[j] becomes a large sentinel value.
    # N + 1 is sufficient, as any real distance is at most N - 1.
    D = [(match_idx[j] - j) if match_idx[j] < N else N + 1 for j in range(N)]

    # 3. Build a Segment Tree for Range Maximum Query (RMQ) on array D.
    seg_tree = [0] * (4 * N)
    
    def build_seg_tree(node, start, end):
        if start == end:
            seg_tree[node] = D[start]
            return
        mid = (start + end) // 2
        build_seg_tree(2 * node, start, mid)
        build_seg_tree(2 * node + 1, mid + 1, end)
        seg_tree[node] = max(seg_tree[2 * node], seg_tree[2 * node + 1])

    if N > 0:
        build_seg_tree(1, 0, N - 1)

    def query_seg_tree(node, start, end, l, r):
        # If query range is invalid or outside the current node's range
        if l > r or r < start or end < l:
            # Return a value that won't affect the maximum in a max() call
            return -1
        # If current node's range is completely within the query range
        if l <= start and end <= r:
            return seg_tree[node]
        
        mid = (start + end) // 2
        p1 = query_seg_tree(2 * node, start, mid, l, r)
        p2 = query_seg_tree(2 * node + 1, mid + 1, end, l, r)
        return max(p1, p2)

    # --- Process Queries ---
    
    results = []
    for _ in range(Q):
        L, R = map(int, input().split())
        # Convert to 0-based indexing for array access
        l, r = L - 1, R - 1
        
        # Binary search for the maximum K.
        low = 1
        high = (r - l + 1) // 2
        ans = 0
        
        while low <= high:
            k = (low + high) // 2
            
            # To form k pairs, we greedily match the k smallest mochi with the
            # k largest mochi in the range [l, r].
            # Condition: max(D[j] for j in [l...l+k-1]) <= (r-l+1-k)
            
            max_D = query_seg_tree(1, 0, N - 1, l, l + k - 1)
            required_dist = r - l + 1 - k
            
            if max_D <= required_dist:
                # k pairs are possible, try for a larger k
                ans = k
                low = k + 1
            else:
                # k pairs are not possible, try for a smaller k
                high = k - 1
        
        results.append(str(ans))
        
    print("
".join(results))

if __name__ == "__main__":
    main()