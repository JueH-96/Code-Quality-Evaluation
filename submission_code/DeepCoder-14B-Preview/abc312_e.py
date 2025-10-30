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
        
        # Ensure x1 < x2, y1 < y2, z1 < z2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        if z1 > z2:
            z1, z2 = z2, z1
        
        cuboids.append((x1, x2, y1, y2, z1, z2))
    
    # Build the dictionaries for each face
    x_front = defaultdict(list)
    x_back = defaultdict(list)
    y_front = defaultdict(list)
    y_back = defaultdict(list)
    z_front = defaultdict(list)
    z_back = defaultdict(list)
    
    for idx in range(N):
        x1, x2, y1, y2, z1, z2 = cuboids[idx]
        x_front[x2].append(idx)
        x_back[x1].append(idx)
        y_front[y2].append(idx)
        y_back[y1].append(idx)
        z_front[z2].append(idx)
        z_back[z1].append(idx)
    
    result = [0] * N
    
    for i in range(N):
        x1i, x2i, y1i, y2i, z1i, z2i = cuboids[i]
        count = 0
        
        # Check x direction: j.x1 == x2i (right)
        for j in x_back.get(x2i, []):
            if j == i:
                continue
            # Check y and z overlaps
            yj1, yj2 = cuboids[j][2], cuboids[j][3]
            y_overlap = (y1i < yj2) and (yj1 < y2i)
            zj1, zj2 = cuboids[j][4], cuboids[j][5]
            z_overlap = (z1i < zj2) and (zj1 < z2i)
            if y_overlap and z_overlap:
                count += 1
        
        # Check x direction: j.x2 == x1i (left)
        for j in x_front.get(x1i, []):
            if j == i:
                continue
            yj1, yj2 = cuboids[j][2], cuboids[j][3]
            y_overlap = (y1i < yj2) and (yj1 < y2i)
            zj1, zj2 = cuboids[j][4], cuboids[j][5]
            z_overlap = (z1i < zj2) and (zj1 < z2i)
            if y_overlap and z_overlap:
                count += 1
        
        # Check y direction: j.y1 == y2i (up)
        for j in y_back.get(y2i, []):
            if j == i:
                continue
            xj1, xj2 = cuboids[j][0], cuboids[j][1]
            x_overlap = (x1i < xj2) and (xj1 < x2i)
            zj1, zj2 = cuboids[j][4], cuboids[j][5]
            z_overlap = (z1i < zj2) and (zj1 < z2i)
            if x_overlap and z_overlap:
                count += 1
        
        # Check y direction: j.y2 == y1i (down)
        for j in y_front.get(y1i, []):
            if j == i:
                continue
            xj1, xj2 = cuboids[j][0], cuboids[j][1]
            x_overlap = (x1i < xj2) and (xj1 < x2i)
            zj1, zj2 = cuboids[j][4], cuboids[j][5]
            z_overlap = (z1i < zj2) and (zj1 < z2i)
            if x_overlap and z_overlap:
                count += 1
        
        # Check z direction: j.z1 == z2i (front)
        for j in z_back.get(z2i, []):
            if j == i:
                continue
            xj1, xj2 = cuboids[j][0], cuboids[j][1]
            x_overlap = (x1i < xj2) and (xj1 < x2i)
            yj1, yj2 = cuboids[j][2], cuboids[j][3]
            y_overlap = (y1i < yj2) and (yj1 < y2i)
            if x_overlap and y_overlap:
                count += 1
        
        # Check z direction: j.z2 == z1i (back)
        for j in z_front.get(z1i, []):
            if j == i:
                continue
            xj1, xj2 = cuboids[j][0], cuboids[j][1]
            x_overlap = (x1i < xj2) and (xj1 < x2i)
            yj1, yj2 = cuboids[j][2], cuboids[j][3]
            y_overlap = (y1i < yj2) and (yj1 < y2i)
            if x_overlap and y_overlap:
                count += 1
        
        result[i] = count
    
    for res in result:
        print(res)

if __name__ == '__main__':
    main()