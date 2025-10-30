# YOUR CODE HERE
def solve():
    n, h, w = map(int, input().split())
    tiles = []
    for _ in range(n):
        tiles.append(list(map(int, input().split())))

    def check(arrangement):
        grid = [[0 for _ in range(w)] for _ in range(h)]
        area_covered = 0
        for i in range(len(arrangement)):
            tile_index, rot, r, c = arrangement[i]
            a, b = tiles[tile_index][0], tiles[tile_index][1]
            if rot:
                a, b = b, a
            
            valid_placement = True
            for x in range(r, r + a):
                for y in range(c, c + b):
                    if x < 0 or x >= h or y < 0 or y >= w or grid[x][y] != 0:
                        valid_placement = False
                        break
                if not valid_placement:
                    break
            
            if valid_placement:
                for x in range(r, r + a):
                    for y in range(c, c + b):
                        grid[x][y] = 1
                area_covered += a * b

        return area_covered == h * w

    import itertools
    
    for i in range(1 << n):
        selected_tiles = []
        for j in range(n):
            if (i >> j) & 1:
                selected_tiles.append(j)

        if not selected_tiles:
            continue

        for arrangement_tuple in itertools.product(*[[(tile_index, rot, r, c) for rot in [0,1] for r in range(h+1) for c in range(w+1)] for tile_index in selected_tiles]):
            if check(list(arrangement_tuple)):
                print("Yes")
                return
    print("No")

solve()