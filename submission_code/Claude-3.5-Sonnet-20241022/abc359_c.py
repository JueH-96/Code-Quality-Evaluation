def get_tile_id(x, y):
    # Returns the unique identifier for the tile containing point (x,y)
    if (x + y) % 2 == 0:
        return (x - x%2, y)
    else:
        return (x - (x+1)%2, y)

def solve():
    Sx, Sy = map(int, input().split())
    Tx, Ty = map(int, input().split())
    
    # Add 0.5 to coordinates as per problem
    start_tile = get_tile_id(Sx, Sy)
    end_tile = get_tile_id(Tx, Ty)
    
    if start_tile == end_tile:
        return 0
        
    # Count unique tiles crossed
    visited = {start_tile}
    curr_x, curr_y = Sx, Sy
    ans = 0
    
    # Move horizontally first
    while curr_x != Tx:
        if curr_x < Tx:
            curr_x += 1
        else:
            curr_x -= 1
        tile = get_tile_id(curr_x, curr_y)
        if tile not in visited:
            visited.add(tile)
            ans += 1
            
    # Then move vertically
    while curr_y != Ty:
        if curr_y < Ty:
            curr_y += 1
        else:
            curr_y -= 1
        tile = get_tile_id(curr_x, curr_y)
        if tile not in visited:
            visited.add(tile)
            ans += 1
            
    return ans

print(solve())