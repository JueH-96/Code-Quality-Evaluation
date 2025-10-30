# YOUR CODE HERE
def solve():
    sx, sy = map(int, input().split())
    tx, ty = map(int, input().split())

    def get_tile(x, y):
        return (x + y) // 2

    start_tile_x = get_tile(sx, sy)
    start_tile_y = get_tile(sx + 1, sy)
    
    end_tile_x = get_tile(tx, ty)
    end_tile_y = get_tile(tx+1, ty)
    
    if start_tile_x == end_tile_x and start_tile_y == end_tile_y:
        print(0)
    elif start_tile_x == end_tile_x or start_tile_y == end_tile_y or abs(start_tile_x - end_tile_x) == abs(start_tile_y - end_tile_y) :
        print(abs(start_tile_x - end_tile_x) + abs(start_tile_y - end_tile_y))
    
    else:
        print(max(abs(start_tile_x - end_tile_x), abs(start_tile_y - end_tile_y)))


solve()