import sys

def main():
    """
    Solves the grid wall destruction problem using Disjoint Set Union (DSU).
    """
    
    # It's safer to use an iterative find to avoid recursion depth issues on large, skewed grids.
    # Python's default recursion limit is ~1000, while H or W can be up to 4e5.
    # An iterative find is robust against worst-case inputs for DSU.
    
    # Read H, W, Q from the first line
    try:
        line = sys.stdin.readline()
        if not line: return
        H, W, Q = map(int, line.split())
    except (IOError, ValueError):
        return

    # A 2D list is faster than a set for grid lookups here.
    has_wall = [[True] * W for _ in range(H)]

    # Disjoint Set Union (DSU) parent arrays for each of the four directions.
    
    # For finding the next wall to the RIGHT in each row.
    # Elements are 0..W, where W is a sentinel for "off the grid".
    pr = [[j for j in range(W + 1)] for _ in range(H)]
    
    # For finding the next wall to the LEFT in each row.
    # Elements are 0..W, mapping columns -1..W-1 to 0..W.
    # 0 is a sentinel for "off the grid left". Column c maps to index c+1.
    pl = [[j for j in range(W + 1)] for _ in range(H)]
    
    # For finding the next wall DOWN in each column.
    # Elements are 0..H, where H is a sentinel for "off the grid".
    pd = [[i for i in range(H + 1)] for _ in range(W)]
    
    # For finding the next wall UP in each column.
    # Elements are 0..H, mapping rows -1..H-1 to 0..H.
    # 0 is a sentinel for "off the grid up". Row r maps to index r+1.
    pu = [[i for i in range(H + 1)] for _ in range(W)]

    # DSU find operation (iterative with path compression).
    def find(parent_list, i):
        path = []
        curr = i
        while parent_list[curr] != curr:
            path.append(curr)
            curr = parent_list[curr]
        root = curr
        for node in path:
            parent_list[node] = root
        return root

    # DSU union operation. The direction of union matters for our logic.
    def union(parent_list, i, j):
        root_i = find(parent_list, i)
        root_j = find(parent_list, j)
        if root_i != root_j:
            parent_list[root_i] = root_j

    total_walls = H * W

    # Process each query
    for line in sys.stdin:
        R, C = map(int, line.split())
        r, c = R - 1, C - 1  # Convert to 0-indexed
        
        to_destroy = []
        if has_wall[r][c]:
            to_destroy.append((r, c))
        else:
            # Bomb at an empty cell: find up to 4 targets simultaneously.
            
            # Find wall to the right
            cr_root = find(pr[r], c + 1)
            if cr_root < W: to_destroy.append((r, cr_root))
            
            # Find wall to the left (column c-1 maps to DSU index c)
            kl_root = find(pl[r], c)
            cl = kl_root - 1
            if cl >= 0: to_destroy.append((r, cl))

            # Find wall down
            rd_root = find(pd[c], r + 1)
            if rd_root < H: to_destroy.append((rd_root, c))
            
            # Find wall up (row r-1 maps to DSU index r)
            ku_root = find(pu[c], r)
            ru = ku_root - 1
            if ru >= 0: to_destroy.append((ru, c))

        # Destroy the collected targets
        for r_d, c_d in to_destroy:
            if has_wall[r_d][c_d]:
                has_wall[r_d][c_d] = False
                total_walls -= 1
                
                # Update all four DSU structures for the destroyed wall
                union(pr[r_d], c_d, c_d + 1)
                union(pl[r_d], c_d + 1, c_d)
                union(pd[c_d], r_d, r_d + 1)
                union(pu[c_d], r_d + 1, r_d)

    print(total_walls)

if __name__ == "__main__":
    main()