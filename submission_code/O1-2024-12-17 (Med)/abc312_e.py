def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast parsing of input
    N = int(input_data[0])
    coords = input_data[1:]
    # Each cuboid: (x1, y1, z1, x2, y2, z2)

    # ----------------------------------------------------------------
    # Special case: if N == 1, the answer is just 0
    # because there are no other cuboids to share a face with.
    # ----------------------------------------------------------------
    if N == 1:
        print(0)
        return

    # coverage[x*10000 + y*100 + z] will store the index of the cuboid
    # that occupies the cell (x,y,z), or -1 if no cuboid occupies it.
    # We have a 100x100x100 grid (x in [0..99], y in [0..99], z in [0..99]).
    # That is 1 million cells in total.
    coverage = [-1] * (100 * 100 * 100)

    # Helper to convert (x,y,z) -> linear index
    def idx(x, y, z):
        return x*10000 + y*100 + z

    # Read and fill the coverage array
    # We parse the cuboid input
    pos = 0
    cuboids = []
    for i in range(N):
        x1 = int(coords[pos]);   y1 = int(coords[pos+1]);   z1 = int(coords[pos+2])
        x2 = int(coords[pos+3]); y2 = int(coords[pos+4]);   z2 = int(coords[pos+5])
        pos += 6
        cuboids.append((x1,y1,z1,x2,y2,z2))

    # Fill coverage for each cuboid
    # Because the cuboids are disjoint in volume, we can safely fill.
    for i, (x1,y1,z1,x2,y2,z2) in enumerate(cuboids):
        # x2,y2,z2 are strictly greater than x1,y1,z1, so volume >= 1
        for x in range(x1, x2):
            base = x*10000
            for y in range(y1, y2):
                rowstart = base + y*100
                for z in range(z1, z2):
                    coverage[rowstart + z] = i

    # Now we detect face-sharing by checking adjacent cells in x, y, z directions.
    # Two adjacent cells in x-direction share a face in the plane x+1/ x if they belong
    # to different cuboids (both >= 0).
    # We collect all distinct pairs (i, j) with i < j that share at least one face.

    pairs = set()
    # Check adjacency in x-direction
    # We'll iterate x in [0..98], compare coverage of (x,y,z) and (x+1,y,z).
    for x in range(99):
        bx = x * 10000
        bxn = bx + 10000  # for x+1
        for y in range(100):
            rowstart = bx + y*100
            rowstart_next = bxn + y*100
            for z in range(100):
                i1 = coverage[rowstart + z]
                i2 = coverage[rowstart_next + z]
                if i1 >= 0 and i2 >= 0 and i1 != i2:
                    if i1 < i2:
                        pairs.add((i1, i2))
                    else:
                        pairs.add((i2, i1))

    # Check adjacency in y-direction
    # Iterate y in [0..98], compare coverage of (x,y,z) and (x,y+1,z).
    for x in range(100):
        bx = x * 10000
        for y in range(99):
            rowstart = bx + y*100
            rowstart_next = rowstart + 100  # y+1
            for z in range(100):
                i1 = coverage[rowstart + z]
                i2 = coverage[rowstart_next + z]
                if i1 >= 0 and i2 >= 0 and i1 != i2:
                    if i1 < i2:
                        pairs.add((i1, i2))
                    else:
                        pairs.add((i2, i1))

    # Check adjacency in z-direction
    # Iterate z in [0..98], compare coverage of (x,y,z) and (x,y,z+1).
    for x in range(100):
        bx = x * 10000
        for y in range(100):
            rowstart = bx + y*100
            for z in range(99):
                i1 = coverage[rowstart + z]
                i2 = coverage[rowstart + z + 1]
                if i1 >= 0 and i2 >= 0 and i1 != i2:
                    if i1 < i2:
                        pairs.add((i1, i2))
                    else:
                        pairs.add((i2, i1))

    # Now count how many distinct neighbors each cuboid has
    adjacency_count = [0]*N
    for (c1, c2) in pairs:
        adjacency_count[c1] += 1
        adjacency_count[c2] += 1

    # Output
    # For each cuboid, print the number of other cuboids that share a face
    out = []
    for i in range(N):
        out.append(str(adjacency_count[i]))
    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()