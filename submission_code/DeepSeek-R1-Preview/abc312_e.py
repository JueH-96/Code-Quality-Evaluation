import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    cuboids = []
    for idx in range(N):
        parts = list(map(int, sys.stdin.readline().split()))
        x1, y1, z1, x2, y2, z2 = parts
        cuboids.append((x1, y1, z1, x2, y2, z2, idx))
    
    x_dict = defaultdict(list)
    y_dict = defaultdict(list)
    z_dict = defaultdict(list)
    
    for cuboid in cuboids:
        x1, y1, z1, x2, y2, z2, idx = cuboid
        key = (y1, y2, z1, z2)
        x_dict[key].append((x1, x2, idx))
        key = (x1, x2, z1, z2)
        y_dict[key].append((y1, y2, idx))
        key = (x1, x2, y1, y2)
        z_dict[key].append((z1, z2, idx))
    
    counts = [0] * N
    
    # Process x_dict groups
    for key in x_dict:
        group = x_dict[key]
        x1_count = defaultdict(int)
        x2_count = defaultdict(int)
        for x1, x2, idx in group:
            x1_count[x1] += 1
            x2_count[x2] += 1
        for x1, x2, idx in group:
            cnt = x1_count.get(x2, 0) + x2_count.get(x1, 0)
            counts[idx] += cnt
    
    # Process y_dict groups
    for key in y_dict:
        group = y_dict[key]
        y1_count = defaultdict(int)
        y2_count = defaultdict(int)
        for y1, y2, idx in group:
            y1_count[y1] += 1
            y2_count[y2] += 1
        for y1, y2, idx in group:
            cnt = y1_count.get(y2, 0) + y2_count.get(y1, 0)
            counts[idx] += cnt
    
    # Process z_dict groups
    for key in z_dict:
        group = z_dict[key]
        z1_count = defaultdict(int)
        z2_count = defaultdict(int)
        for z1, z2, idx in group:
            z1_count[z1] += 1
            z2_count[z2] += 1
        for z1, z2, idx in group:
            cnt = z1_count.get(z2, 0) + z2_count.get(z1, 0)
            counts[idx] += cnt
    
    for c in counts:
        print(c)

if __name__ == '__main__':
    main()