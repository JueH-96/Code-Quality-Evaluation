class Cuboid:
    def __init__(self, index, x1, x2, y1, y2, z1, z2):
        self.index = index
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2

def process_group(cuboids, counts, coord1, coord2):
    grid = [[0] * 101 for _ in range(101)]
    for cuboid in cuboids:
        c1 = getattr(cuboid, coord1 + '_1')
        c2 = getattr(cuboid, coord1 + '_2')
        c3 = getattr(cuboid, coord2 + '_1')
        c4 = getattr(cuboid, coord2 + '_2')
        grid[c1][c3] += 1
        if c2 + 1 < 101:
            grid[c2 + 1][c3] -= 1
        if c4 + 1 < 101:
            grid[c1][c4 + 1] -= 1
        if c2 + 1 < 101 and c4 + 1 < 101:
            grid[c2 + 1][c4 + 1] += 1
    for i in range(1, 101):
        for j in range(101):
            grid[i][j] += grid[i - 1][j]
    for j in range(1, 101):
        for i in range(101):
            grid[i][j] += grid[i][j - 1]
    for cuboid in cuboids:
        c1 = getattr(cuboid, coord1 + '_1')
        c2 = getattr(cuboid, coord1 + '_2')
        c3 = getattr(cuboid, coord2 + '_1')
        c4 = getattr(cuboid, coord2 + '_2')
        total = grid[c2][c4] - (grid[c1 - 1][c4] if c1 - 1 >= 0 else 0) - (grid[c2][c3 - 1] if c3 - 1 >= 0 else 0) + (grid[c1 - 1][c3 - 1] if c1 - 1 >= 0 and c3 - 1 >= 0 else 0)
        count = total - 1
        counts[cuboid.index] += count

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    cuboids = []
    idx = 1
    for i in range(N):
        x1 = int(data[idx])
        y1 = int(data[idx + 1])
        z1 = int(data[idx + 2])
        x2 = int(data[idx + 3])
        y2 = int(data[idx + 4])
        z2 = int(data[idx + 5])
        cuboids.append(Cuboid(i, x1, x2, y1, y2, z1, z2))
        idx += 6
    counts = [0] * N
    directions = [
        {'group_key': 'z2', 'coords': ('x', 'y')},
        {'group_key': 'z1', 'coords': ('x', 'y')},
        {'group_key': 'x1', 'coords': ('y', 'z')},
        {'group_key': 'x2', 'coords': ('y', 'z')},
        {'group_key': 'y1', 'coords': ('x', 'z')},
        {'group_key': 'y2', 'coords': ('x', 'z')}
    ]
    for direction in directions:
        group_dict = {}
        for cuboid in cuboids:
            key = getattr(cuboid, direction['group_key'])
            if key not in group_dict:
                group_dict[key] = []
            group_dict[key].append(cuboid)
        for group in group_dict.values():
            coord1, coord2 = direction['coords']
            process_group(group, counts, coord1, coord2)
    for i in range(N):
        print(counts[i])

if __name__ == '__main__':
    main()