import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    cuboids = []
    for _ in range(N):
        x1 = int(input[ptr])
        y1 = int(input[ptr+1])
        z1 = int(input[ptr+2])
        x2 = int(input[ptr+3])
        y2 = int(input[ptr+4])
        z2 = int(input[ptr+5])
        ptr += 6
        cuboids.append((x1, y1, z1, x2, y2, z2))
    
    # Initialize dictionaries for start and end of each axis
    x_start = defaultdict(list)
    x_end = defaultdict(list)
    y_start = defaultdict(list)
    y_end = defaultdict(list)
    z_start = defaultdict(list)
    z_end = defaultdict(list)
    
    for idx in range(N):
        x1, y1, z1, x2, y2, z2 = cuboids[idx]
        x_start[x1].append(idx)
        x_end[x2].append(idx)
        y_start[y1].append(idx)
        y_end[y2].append(idx)
        z_start[z1].append(idx)
        z_end[z2].append(idx)
    
    ans = [0] * N
    
    for i in range(N):
        xi1, yi1, zi1, xi2, yi2, zi2 = cuboids[i]
        
        # Process x direction
        # Case 1: j's x_start is i's x_end
        for j in x_start.get(xi2, []):
            if i < j:
                xj1, yj1, zj1, xj2, yj2, zj2 = cuboids[j]
                y_overlap = max(yi1, yj1) < min(yi2, yj2)
                z_overlap = max(zi1, zj1) < min(zi2, zj2)
                if y_overlap and z_overlap:
                    ans[i] += 1
                    ans[j] += 1
        # Case 2: j's x_end is i's x_start
        for j in x_end.get(xi1, []):
            if i < j:
                xj1, yj1, zj1, xj2, yj2, zj2 = cuboids[j]
                y_overlap = max(yi1, yj1) < min(yi2, yj2)
                z_overlap = max(zi1, zj1) < min(zi2, zj2)
                if y_overlap and z_overlap:
                    ans[i] += 1
                    ans[j] += 1
        
        # Process y direction
        # Case 1: j's y_start is i's y_end
        for j in y_start.get(yi2, []):
            if i < j:
                xj1, yj1, zj1, xj2, yj2, zj2 = cuboids[j]
                x_overlap = max(xi1, xj1) < min(xi2, xj2)
                z_overlap = max(zi1, zj1) < min(zi2, zj2)
                if x_overlap and z_overlap:
                    ans[i] += 1
                    ans[j] += 1
        # Case 2: j's y_end is i's y_start
        for j in y_end.get(yi1, []):
            if i < j:
                xj1, yj1, zj1, xj2, yj2, zj2 = cuboids[j]
                x_overlap = max(xi1, xj1) < min(xi2, xj2)
                z_overlap = max(zi1, zj1) < min(zi2, zj2)
                if x_overlap and z_overlap:
                    ans[i] += 1
                    ans[j] += 1
        
        # Process z direction
        # Case 1: j's z_start is i's z_end
        for j in z_start.get(zi2, []):
            if i < j:
                xj1, yj1, zj1, xj2, yj2, zj2 = cuboids[j]
                x_overlap = max(xi1, xj1) < min(xi2, xj2)
                y_overlap = max(yi1, yj1) < min(yi2, yj2)
                if x_overlap and y_overlap:
                    ans[i] += 1
                    ans[j] += 1
        # Case 2: j's z_end is i's z_start
        for j in z_end.get(zi1, []):
            if i < j:
                xj1, yj1, zj1, xj2, yj2, zj2 = cuboids[j]
                x_overlap = max(xi1, xj1) < min(xi2, xj2)
                y_overlap = max(yi1, yj1) < min(yi2, yj2)
                if x_overlap and y_overlap:
                    ans[i] += 1
                    ans[j] += 1
    
    for a in ans:
        print(a)

if __name__ == "__main__":
    main()