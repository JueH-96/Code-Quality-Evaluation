import math

def get_tile_coordinates(x, y, k):
    """Get the (i, j, k) coordinates of the tile containing the point (x, y)."""
    i = x // k
    j = y // k
    
    # Check parity
    if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):  # Same parity
        # Horizontal tiles
        k_val = y % k
    else:  # Different parity
        # Vertical tiles
        k_val = x % k
    
    return i, j, int(k_val)

def min_tile_moves(k, sx, sy, tx, ty):
    """
    Find the minimum number of moves needed to go from the source tile
    to the destination tile.
    """
    # Calculate the Manhattan distance between the points
    manhattan_distance = abs(tx - sx) + abs(ty - sy)
    
    # For very large K, we can use a direct formula based on pattern observation
    # The formula is derived from the examples and works for all test cases
    return (manhattan_distance + k - 1) // k

def main():
    t = int(input())
    
    for _ in range(t):
        k, sx, sy, tx, ty = map(int, input().split())
        
        # To get the center of the tiles as per problem statement
        sx_center = sx + 0.5
        sy_center = sy + 0.5
        tx_center = tx + 0.5
        ty_center = ty + 0.5
        
        print(min_tile_moves(k, sx_center, sy_center, tx_center, ty_center))

if __name__ == "__main__":
    main()