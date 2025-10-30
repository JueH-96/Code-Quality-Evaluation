def calculate_toll_horizontal_first(sx, sy, tx, ty):
    # Horizontal movement first
    horiz_toll = 0
    vert_toll = 0
    
    # Calculate horizontal toll
    if sx == tx:
        # No horizontal movement needed
        horiz_toll = 0
    elif sx < tx:
        # Moving right
        if (sx + sy) % 2 == 0:  # Starting on a horizontal tile
            horiz_toll = (tx - sx) // 2
        else:  # Starting on a vertical tile
            horiz_toll = (tx - sx + 1) // 2
    else:  # sx > tx
        # Moving left
        if (sx + sy) % 2 == 1:  # Starting on a vertical tile
            horiz_toll = (sx - tx) // 2
        else:  # Starting on a horizontal tile
            horiz_toll = (sx - tx + 1) // 2
    
    # Calculate vertical toll (after horizontal movement)
    if sy == ty:
        # No vertical movement needed
        vert_toll = 0
    elif sy < ty:
        # Moving up
        if (tx + sy) % 2 == 1:  # Starting on a vertical tile after horizontal movement
            vert_toll = (ty - sy) // 2
        else:  # Starting on a horizontal tile after horizontal movement
            vert_toll = (ty - sy + 1) // 2
    else:  # sy > ty
        # Moving down
        if (tx + sy) % 2 == 0:  # Starting on a horizontal tile after horizontal movement
            vert_toll = (sy - ty) // 2
        else:  # Starting on a vertical tile after horizontal movement
            vert_toll = (sy - ty + 1) // 2
    
    return horiz_toll + vert_toll

def calculate_toll_vertical_first(sx, sy, tx, ty):
    # Vertical movement first
    vert_toll = 0
    horiz_toll = 0
    
    # Calculate vertical toll
    if sy == ty:
        # No vertical movement needed
        vert_toll = 0
    elif sy < ty:
        # Moving up
        if (sx + sy) % 2 == 1:  # Starting on a vertical tile
            vert_toll = (ty - sy) // 2
        else:  # Starting on a horizontal tile
            vert_toll = (ty - sy + 1) // 2
    else:  # sy > ty
        # Moving down
        if (sx + sy) % 2 == 0:  # Starting on a horizontal tile
            vert_toll = (sy - ty) // 2
        else:  # Starting on a vertical tile
            vert_toll = (sy - ty + 1) // 2
    
    # Calculate horizontal toll (after vertical movement)
    if sx == tx:
        # No horizontal movement needed
        horiz_toll = 0
    elif sx < tx:
        # Moving right
        if (sx + ty) % 2 == 0:  # Starting on a horizontal tile after vertical movement
            horiz_toll = (tx - sx) // 2
        else:  # Starting on a vertical tile after vertical movement
            horiz_toll = (tx - sx + 1) // 2
    else:  # sx > tx
        # Moving left
        if (sx + ty) % 2 == 1:  # Starting on a vertical tile after vertical movement
            horiz_toll = (sx - tx) // 2
        else:  # Starting on a horizontal tile after vertical movement
            horiz_toll = (sx - tx + 1) // 2
    
    return vert_toll + horiz_toll

def solve():
    sx, sy = map(int, input().split())
    tx, ty = map(int, input().split())
    
    # Calculate toll for both movement strategies
    toll_horizontal_first = calculate_toll_horizontal_first(sx, sy, tx, ty)
    toll_vertical_first = calculate_toll_vertical_first(sx, sy, tx, ty)
    
    # Return minimum toll
    return min(toll_horizontal_first, toll_vertical_first)

print(solve())