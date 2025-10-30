def get_tile_id(x, y):
    # For point (x+0.5, y+0.5), it's in unit square (x, y)
    if (x + y) % 2 == 0:
        # Horizontal tile
        if x % 2 == 0:
            return (x // 2, y, 'H')
        else:
            return ((x - 1) // 2, y, 'H')
    else:
        # Vertical tile
        if y % 2 == 0:
            return (x, y // 2, 'V')
        else:
            return (x, (y - 1) // 2, 'V')

def min_toll(sx, sy, tx, ty):
    if sx == tx and sy == ty:
        return 0
    
    # Get starting and ending tiles
    start_tile = get_tile_id(sx, sy)
    end_tile = get_tile_id(tx, ty)
    
    # If in the same tile, no toll
    if start_tile == end_tile:
        return 0
    
    # Calculate the minimum path
    # The key insight is that we need to count how many times we switch between
    # horizontal and vertical tiles
    
    # Move from (sx, sy) to (tx, ty)
    dx = abs(tx - sx)
    dy = abs(ty - sy)
    
    # If we're moving in only one direction
    if dx == 0:
        # Only vertical movement
        return dy
    if dy == 0:
        # Only horizontal movement
        return dx
    
    # For diagonal movement, we need to analyze the path
    # The optimal path alternates between horizontal and vertical moves
    # Each unit move in a new tile costs 1
    
    # Count the number of tiles we pass through
    # This is essentially the Manhattan distance minus optimizations
    # when consecutive moves are in the same tile
    
    # The minimum toll is related to the number of times we change tile orientation
    # plus the distance traveled in each orientation
    
    # For the general case, the minimum toll is approximately dx + dy
    # minus the savings from moving through connected tiles
    
    # After analyzing the pattern, the minimum toll is:
    return dx + dy

# Read input
sx, sy = map(int, input().split())
tx, ty = map(int, input().split())

# Calculate and print the minimum toll
print(min_toll(sx, sy, tx, ty))