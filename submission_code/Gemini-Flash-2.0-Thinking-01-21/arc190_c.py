import sys

# Set recursion depth higher if needed
# sys.setrecursionlimit(2000)

MOD = 998244353

def add(a, b):
    res = (a + b) % MOD
    if res < 0:
        res += MOD
    return res

def mul(a, b):
    return (a * b) % MOD

def solve():
    H, W = map(int, sys.stdin.readline().split())
    A = []
    for _ in range(H):
        A.append(list(map(int, sys.stdin.readline().split())))

    Q, sh, sw = map(int, sys.stdin.readline().split())

    # Adjust to 0-indexed for implementation
    sh -= 1
    sw -= 1

    # DP tables
    # dp_f[h][w]: sum of products from (0,0) to (h,w)
    # dp_b[h][w]: sum of products from (h,w) to (H-1, W-1)
    dp_f = [[0 for _ in range(W)] for _ in range(H)]
    dp_b = [[0 for _ in range(W)] for _ in range(H)]

    # Initial DP calculation for dp_f
    dp_f[0][0] = A[0][0]
    for h in range(H):
        for w in range(W):
            if h == 0 and w == 0: continue
            from_up = dp_f[h-1][w] if h > 0 else 0
            from_left = dp_f[h][w-1] if w > 0 else 0
            dp_f[h][w] = mul(add(from_up, from_left), A[h][w])

    # Initial DP calculation for dp_b
    dp_b[H-1][W-1] = A[H-1][W-1]
    for h in range(H-1, -1, -1):
        for w in range(W-1, -1, -1):
            if h == H-1 and w == W-1: continue
            from_down = dp_b[h+1][w] if h < H-1 else 0
            from_right = dp_b[h][w+1] if w < W-1 else 0
            dp_b[h][w] = mul(add(from_down, from_right), A[h][w])

    # Current total sum
    current_sum = dp_f[H-1][W-1]

    # DP* calculation helper (sum of products excluding current cell)
    # Based on DP values BEFORE the update to A[r][c]
    # This calculates the sum of products for paths through (r,c), EXCLUDING the value at (r,c)
    # It's the sum of products of paths from (0,0) to (r,c-1) + sum of products of paths from (0,0) to (r-1,c)
    # combined with paths from (r,c+1) + paths from (r+1,c) to (H-1,W-1)
    def get_dpf_star(r, c):
        # sum of products along paths from (0,0) to (r,c), excluding A[r][c]
        # Path comes from (r-1,c) or (r,c-1). Sum of products up to that point.
        # DP_f(r-1,c) is the sum of products along paths from (0,0) to (r-1,c)
        # DP_f(r,c-1) is the sum of products along paths from (0,0) to (r,c-1)
        # Any path to (r,c) is a path to (r-1,c) followed by Down or path to (r,c-1) followed by Right.
        # The sum of products on paths ending at (r,c), *excluding* A[r][c], is the sum of products up to the cell BEFORE (r,c).
        # The sum of products up to (r-1,c) is dp_f[r-1][c].
        # The sum of products up to (r,c-1) is dp_f[r][c-1].
        # So the sum of products on paths to (r,c), excluding A[r][c], is the sum of DP_f of the predecessors.
        
        from_up_sum = dp_f[r-1][c] if r > 0 else 0
        from_left_sum = dp_f[r][c-1] if c > 0 else 0
        
        if r == 0 and c == 0: return 1 # Path to (0,0) excluding A[0][0]. Path is just (0,0). Product is 1 if A[0][0] is excluded.
        
        return add(from_up_sum, from_left_sum)

    def get_dpb_star(r, c):
        # sum of products along paths from (r,c) to (H-1,W-1), excluding A[r][c]
        # Path goes to (r+1,c) or (r,c+1). Sum of products from that point.
        # DP_b(r+1,c) is sum of products from (r+1,c).
        # DP_b(r,c+1) is sum of products from (r,c+1).
        # Any path from (r,c) starts with Down to (r+1,c) or Right to (r,c+1).
        # The sum of products on paths starting at (r,c), *excluding* A[r][c], is the sum of DP_b of the successors.

        from_down_sum = dp_b[r+1][c] if r < H-1 else 0
        from_right_sum = dp_b[r][c+1] if c < W-1 else 0

        if r == H-1 and c == W-1: return 1 # Path from (H-1, W-1) excluding A[H-1][W-1]. Path is just (H-1,W-1). Product is 1 if A[H-1][W-1] is excluded.

        return add(from_down_sum, from_right_sum)


    for _ in range(Q):
        d, a = sys.stdin.readline().split()
        a = int(a)

        # New position (nh, nw)
        nh, nw = sh, sw
        if d == 'U': nh -= 1
        elif d == 'D': nh += 1
        elif d == 'L': nw -= 1
        elif d == 'R': nw += 1

        # Get DP* values BEFORE updating A[nh][nw] and DP tables
        dpf_star_nhnw = get_dpf_star(nh, nw)
        dpb_star_nhnw = get_dpb_star(nh, nw)

        # Sum of products for paths passing through (nh, nw) excluding A[nh][nw]
        sum_paths_through_nhnw_no_AnhNw = mul(dpf_star_nhnw, dpb_star_nhnw)

        # Change in total sum
        old_A = A[nh][nw]
        new_A = a
        
        delta_A = add(new_A, -old_A) # This calculates (new_A - old_A) mod M
        delta_S = mul(sum_paths_through_nhnw_no_AnhNw, delta_A)

        current_sum = add(current_sum, delta_S)

        # Update grid
        A[nh][nw] = new_A

        # Update DP tables by recomputing affected regions
        # Affected region for dp_f: cells (r,c) with r >= nh, c >= nw
        # Recompute dp_f from (nh, nw) to (H-1, W-1)
        for r in range(nh, H):
            for c in range(nw, W):
                if r == 0 and c == 0:
                    dp_f[r][c] = A[r][c]
                    continue
                from_up = dp_f[r-1][c] if r > 0 else 0
                from_left = dp_f[r][c-1] if c > 0 else 0
                dp_f[r][c] = mul(add(from_up, from_left), A[r][c])

        # Affected region for dp_b: cells (r,c) with r <= nh, c <= nw
        # Recompute dp_b from (nh, nw) to (0, 0)
        for r in range(nh, -1, -1):
            for c in range(nw, -1, -1):
                if r == H-1 and c == W-1:
                    dp_b[r][c] = A[r][c]
                    continue
                from_down = dp_b[r+1][c] if r < H-1 else 0
                from_right = dp_b[r][c+1] if c < W-1 else 0
                dp_b[r][c] = mul(add(from_down, from_right), A[r][c])

        # Update Takahashi's position
        sh, sw = nh, nw

        print(current_sum)

solve()