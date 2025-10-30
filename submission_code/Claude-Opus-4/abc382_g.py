def find_tile(x, y, K):
    """Find the tile (i, j, k) containing point (x + 0.5, y + 0.5)"""
    i = x // K
    j = y // K
    
    if (i + j) % 2 == 0:  # Same parity - horizontal strips
        # Tile covers iK ≤ x ≤ (i+1)K and jK + k ≤ y ≤ jK + k + 1
        k = y - j * K
    else:  # Different parity - vertical strips
        # Tile covers iK + k ≤ x ≤ iK + k + 1 and jK ≤ y ≤ (j+1)K
        k = x - i * K
    
    return i, j, k

def min_moves(K, sx, sy, tx, ty):
    """Find minimum moves between tiles containing the two points"""
    si, sj, sk = find_tile(sx, sy, K)
    ti, tj, tk = find_tile(tx, ty, K)
    
    if si == ti and sj == tj:
        # Same block
        return abs(tk - sk)
    
    # Different blocks - need to consider the path
    # The minimum path involves moving through blocks optimally
    
    # Calculate block distance
    di = abs(ti - si)
    dj = abs(tj - sj)
    
    # When moving between blocks, we need to consider:
    # 1. Moving to the edge of current block
    # 2. Moving through intermediate blocks
    # 3. Moving from edge to target position in final block
    
    # For blocks with alternating horizontal/vertical patterns,
    # the minimum moves is related to the Manhattan distance
    # but adjusted for the tile structure
    
    # Base movement between blocks
    moves = di + dj
    
    # Additional moves needed due to tile positions
    if (si + sj) % 2 == 0:  # Starting block has horizontal strips
        # Need to reach appropriate edge
        if di > 0:
            # Moving horizontally, consider vertical position
            moves += min(sk, K - 1 - sk)
    else:  # Starting block has vertical strips
        # Need to reach appropriate edge
        if dj > 0:
            # Moving vertically, consider horizontal position
            moves += min(sk, K - 1 - sk)
    
    if (ti + tj) % 2 == 0:  # Target block has horizontal strips
        # Need to reach target from edge
        if di > 0:
            moves += min(tk, K - 1 - tk)
    else:  # Target block has vertical strips
        if dj > 0:
            moves += min(tk, K - 1 - tk)
    
    # Adjust for the actual tile connectivity pattern
    # The formula needs to account for the fact that tiles connect
    # in specific ways at block boundaries
    
    # After analysis, the minimum moves follows a pattern based on
    # the Manhattan distance in blocks plus adjustments for tile positions
    
    # Simplified calculation based on the problem structure
    if di == 0:
        return dj + abs(tk - sk)
    elif dj == 0:
        return di + abs(tk - sk)
    else:
        # General case - need to traverse both dimensions
        # The pattern creates a specific cost structure
        return di + dj + max(0, abs(tk - sk) - min(di, dj))

# Read input
T = int(input())
for _ in range(T):
    K, sx, sy, tx, ty = map(int, input().split())
    print(min_moves(K, sx, sy, tx, ty))