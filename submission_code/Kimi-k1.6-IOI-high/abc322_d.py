import sys

def get_coords(poly):
    coords = []
    for i in range(4):
        for j in range(4):
            if poly[i][j] == '#':
                coords.append((i, j))
    return coords

def normalize(coords):
    if not coords:
        return []
    min_i = min(c[0] for c in coords)
    min_j = min(c[1] for c in coords)
    return [(i - min_i, j - min_j) for (i, j) in coords]

def generate_unique_shapes(coords):
    if not coords:
        return set()
    original = normalize(coords)
    unique = set()
    for angle in [0, 90, 180, 270]:
        rotated = []
        for (i, j) in original:
            if angle == 0:
                ni, nj = i, j
            elif angle == 90:
                ni, nj = j, -i
            elif angle == 180:
                ni, nj = -i, -j
            elif angle == 270:
                ni, nj = -j, i
            rotated.append((ni, nj))
        normalized = normalize(rotated)
        unique.add(frozenset(normalized))
    return unique

def generate_transformations(coords):
    unique_shapes = generate_unique_shapes(coords)
    transformations = []
    for shape in unique_shapes:
        shape_list = list(shape)
        if not shape_list:
            continue
        max_i = max(p[0] for p in shape_list)
        max_j = max(p[1] for p in shape_list)
        for a in range(0, 4 - max_i):
            for b in range(0, 4 - max_j):
                transformed = set()
                for (i, j) in shape_list:
                    ni = a + i
                    nj = b + j
                    transformed.add((ni, nj))
                transformations.append(transformed)
    return transformations

def main():
    lines = [line.strip() for line in sys.stdin]
    poly1 = lines[0:4]
    poly2 = lines[4:8]
    poly3 = lines[8:12]
    
    poly1_coords = get_coords(poly1)
    poly2_coords = get_coords(poly2)
    poly3_coords = get_coords(poly3)
    
    poly1_trans = generate_transformations(poly1_coords)
    poly2_trans = generate_transformations(poly2_coords)
    poly3_trans = generate_transformations(poly3_coords)
    
    found = False
    for s1 in poly1_trans:
        for s2 in poly2_trans:
            if s1.isdisjoint(s2):
                combined = s1.union(s2)
                for s3 in poly3_trans:
                    if combined.isdisjoint(s3):
                        if len(combined.union(s3)) == 16:
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if found:
            break
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()