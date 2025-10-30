def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    cuboids = []
    idx = 1
    for _ in range(N):
        x1, y1, z1, x2, y2, z2 = map(int, input_data[idx:idx+6])
        idx += 6
        cuboids.append((x1, y1, z1, x2, y2, z2))
    
    # We will use a 1D array of size 101*101*101 to store which cuboid occupies each cell.
    # Index function for 3D -> 1D:
    #   idx(x,y,z) = x*(101*101) + y*101 + z
    # The space ranges from [0..100] in each dimension (101 possible coordinates),
    # but cells are effectively [0..100) in each dimension for adjacency checks.
    # A cuboid [x1,x2) x [y1,y2) x [z1,z2) occupies those discrete cells.
    
    LIMIT = 101
    ARR_SIZE = LIMIT * LIMIT * LIMIT
    arr = [0] * ARR_SIZE
    
    def index_3d(x, y, z):
        return x*(LIMIT*LIMIT) + y*LIMIT + z
    
    # Fill the 3D array with the cuboid IDs (1..N). 0 means empty.
    # The problem states no positive-volume overlap, so each cell is occupied by at most one cuboid.
    for i, (x1, y1, z1, x2, y2, z2) in enumerate(cuboids, start=1):
        # For each discrete cell in the cuboid:
        # (x1..x2-1) in the integer grid, similarly for y,z.
        for x in range(x1, x2):
            base_xy = x * (LIMIT*LIMIT)
            for y in range(y1, y2):
                base_xyz = base_xy + y*LIMIT
                for z in range(z1, z2):
                    arr[base_xyz + z] = i
    
    # We'll keep track of adjacency in a list of sets (so each cuboid's neighbors are stored).
    adjacency_sets = [set() for _ in range(N)]
    
    # Now we check faces between adjacent cells in 3 directions: x+1, y+1, z+1.
    # If arr[cell] != 0 and arr[neighbor] != 0 and they differ, they share a face.
    # We'll record adjacency for both.
    
    # Check x-direction neighbors (x+1)
    for x in range(LIMIT-1):  # 0..99
        for y in range(LIMIT):  # 0..100
            for z in range(LIMIT):  # 0..100
                c1 = arr[index_3d(x, y, z)]
                c2 = arr[index_3d(x+1, y, z)]
                if c1 != 0 and c2 != 0 and c1 != c2:
                    adjacency_sets[c1-1].add(c2-1)
                    adjacency_sets[c2-1].add(c1-1)
    
    # Check y-direction neighbors (y+1)
    for x in range(LIMIT):
        for y in range(LIMIT-1):  # 0..99
            for z in range(LIMIT):
                c1 = arr[index_3d(x, y, z)]
                c2 = arr[index_3d(x, y+1, z)]
                if c1 != 0 and c2 != 0 and c1 != c2:
                    adjacency_sets[c1-1].add(c2-1)
                    adjacency_sets[c2-1].add(c1-1)
    
    # Check z-direction neighbors (z+1)
    for x in range(LIMIT):
        for y in range(LIMIT):
            for z in range(LIMIT-1):  # 0..99
                c1 = arr[index_3d(x, y, z)]
                c2 = arr[index_3d(x, y, z+1)]
                if c1 != 0 and c2 != 0 and c1 != c2:
                    adjacency_sets[c1-1].add(c2-1)
                    adjacency_sets[c2-1].add(c1-1)
    
    # Finally, output the number of neighbors for each cuboid in order.
    out = []
    for i in range(N):
        out.append(str(len(adjacency_sets[i])))
    
    print("
".join(out))

def main():
    solve()

# If you want to run the solution directly, you could uncomment below:
# if __name__ == "__main__":
#     main()