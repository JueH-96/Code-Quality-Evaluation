def main():
    import sys
    from bisect import bisect_left, bisect_right

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    cuboids = []
    idx = 1
    for _ in range(N):
        x1, y1, z1, x2, y2, z2 = map(int, data[idx:idx+6])
        cuboids.append({
            'id': _,
            'x1': x1, 'x2': x2,
            'y1': y1, 'y2': y2,
            'z1': z1, 'z2': z2
        })
        idx += 6

    def process_dimension(cuboids, dim):
        counts = [0] * N
        dims = ['x', 'y', 'z']
        cur_dim = dims[dim]
        other_dims = [(dim + 1) % 3, (dim + 2) % 3]
        other_dim1 = dims[other_dims[0]]
        other_dim2 = dims[other_dims[1]]

        # Collect unique coordinates in the specified dimension
        unique_coords = set()
        for c in cuboids:
            unique_coords.add(c[cur_dim + '1'])
            unique_coords.add(c[cur_dim + '2'])
        unique_coords = sorted(unique_coords)

        for coord in unique_coords:
            # Collect cuboids that have a face at this coordinate
            group = []
            for c in cuboids:
                if c[cur_dim + '1'] == coord or c[cur_dim + '2'] == coord:
                    group.append(c)
            # Sort group by the first other dimension
            group.sort(key=lambda c: (c[other_dim1 + '1'], c[other_dim2 + '1']))
            # Initialize a structure to keep track of active ranges in the second other dimension
            active = []
            for c in group:
                # Query how many active ranges overlap with c's second other dimension range
                y1 = c[other_dim2 + '1']
                y2 = c[other_dim2 + '2']
                i = bisect_right(active, y1, key=lambda x: x[1])
                j = bisect_left(active, y2, key=lambda x: x[0])
                count = i - j
                counts[c['id']] += count
                # Add c's y range to the active set
                active.insert(bisect_left(active, (y1, y2)), (y1, y2))
        return counts

    total_counts = [0] * N
    for dim in range(3):
        counts = process_dimension(cuboids, dim)
        for i in range(N):
            total_counts[i] += counts[i]

    print(' '.join(map(str, total_counts)))

if __name__ == '__main__':
    main()