# YOUR CODE HERE
def get_tile_id(x, y):
    if (x + y) % 2 == 0:
        return (x // 2, y)
    else:
        return ((x - 1) // 2, y)

def solve(sx, sy, tx, ty):
    start_tile = get_tile_id(sx, sy)
    end_tile = get_tile_id(tx, ty)
    
    dx = abs(end_tile[0] - start_tile[0])
    dy = abs(end_tile[1] - start_tile[1])
    
    return dx + dy

sx, sy = map(int, input().split())
tx, ty = map(int, input().split())

print(solve(sx, sy, tx, ty))