# YOUR CODE HERE
import sys

def solve():
    """
    This program solves the tile-crossing problem by transforming the grid
    coordinates into a more regular system and calculating the shortest path.
    """
    try:
        sx, sy = map(int, sys.stdin.readline().split())
        tx, ty = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Handle empty input or malformed lines, which can occur in some testing environments.
        return

    # Transform coordinates (i, j) to a tile coordinate system (u, v) where the grid is regular.
    # The transformation is derived from the tiling pattern:
    # - v-coordinate is the row index: v = j
    # - u-coordinate is the tile index within the row: u = (i - j % 2) // 2
    # This aligns the staggered brick pattern into a standard grid.
    vs = sy
    us = (sx - sy % 2) // 2
    
    vt = ty
    ut = (tx - ty % 2) // 2

    # The cost of travel is the shortest path on the tile graph.
    # Moves can be horizontal, vertical, or diagonal (in the transformed space).
    # All elementary moves (crossing one boundary) cost 1.

    # Calculate the required displacement in the (u,v) system.
    du = ut - us
    dv = vt - vs
    
    # If there is no vertical movement, the cost is purely the horizontal distance in the u-grid.
    if dv == 0:
        print(abs(du))
        return

    # The base cost is the number of vertical boundaries crossed.
    cost = abs(dv)

    # During vertical movement, the u-coordinate can change "for free" due to diagonal moves.
    # We calculate the range of this free change.
    v_start = min(vs, vt)
    v_end = max(vs, vt)
    
    # The range of v-coordinates we move through is [v_start, v_end - 1].
    
    # We need to count the number of even and odd rows in this range.
    # Number of odd integers in [a, b] is (b+1)//2 - a//2
    # Number of even integers in [a, b] is b//2 - (a-1)//2
    v_range_start = v_start
    v_range_end = v_end - 1
    num_odd_v = (v_range_end + 1) // 2 - v_range_start // 2
    num_even_v = v_range_end // 2 - (v_range_start - 1) // 2

    u_change_min = 0
    u_change_max = 0

    if dv > 0:  # Moving up (v increases)
        # u can increase by 1 for each odd v, and decrease by 1 for each even v.
        u_change_max = num_odd_v
        u_change_min = -num_even_v
    else:  # Moving down (v decreases)
        # u can increase by 1 for each even v, and decrease by 1 for each odd v.
        u_change_max = num_even_v
        u_change_min = -num_odd_v

    # If the required change 'du' is outside the range achievable through
    # diagonal moves, we must pay for extra pure-horizontal moves.
    if du > u_change_max:
        cost += du - u_change_max
    elif du < u_change_min:
        cost += u_change_min - du
        
    print(cost)

if __name__ == "__main__":
    solve()