import math

def solve():
    sx, sy = map(int, input().split())
    tx, ty = map(int, input().split())

    if sx == tx and sy == ty:
        print(0)
        return

    dx_abs = abs(sx - tx)
    dy_abs = abs(sy - ty)

    ans = 0

    # Number of diagonal moves
    diag_moves = min(dx_abs, dy_abs)
    ans += diag_moves
    
    rem_dx = dx_abs - diag_moves
    rem_dy = dy_abs - diag_moves

    if rem_dy > 0: 
        # Purely vertical movement remaining. rem_dx is 0.
        # Each vertical step costs 1.
        ans += rem_dy
    elif rem_dx > 0: 
        # Purely horizontal movement remaining. rem_dy is 0.
        
        # Determine the x-coordinate where the pure horizontal segment begins.
        # This segment is along the line y = ty.
        x_coord_at_H_segment_start = 0
        if tx > sx: 
            x_coord_at_H_segment_start = sx + diag_moves
        else: # tx < sx
            x_coord_at_H_segment_start = sx - diag_moves
            
        # Parity of (x_coord_at_H_segment_start + ty) determines cost for horizontal moves.
        parity_sum_at_H_start = (x_coord_at_H_segment_start + ty) % 2
        
        cost_rem_dx = 0
        if tx > sx: # Moving right for rem_dx steps
            if parity_sum_at_H_start == 0: # Start_X + Start_Y is even
                cost_rem_dx = rem_dx // 2
            else: # Start_X + Start_Y is odd
                cost_rem_dx = (rem_dx + 1) // 2
        else: # Moving left for rem_dx steps (tx < sx)
            if parity_sum_at_H_start == 1: # Start_X + Start_Y is odd
                cost_rem_dx = rem_dx // 2
            else: # Start_X + Start_Y is even
                cost_rem_dx = (rem_dx + 1) // 2
        ans += cost_rem_dx
        
    print(ans)

solve()