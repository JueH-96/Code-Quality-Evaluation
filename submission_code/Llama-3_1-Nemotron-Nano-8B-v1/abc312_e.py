import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    cuboids = []
    x1_dict = defaultdict(list)
    x2_dict = defaultdict(list)
    y1_dict = defaultdict(list)
    y2_dict = defaultdict(list)
    z1_dict = defaultdict(list)
    z2_dict = defaultdict(list)
    
    for _ in range(N):
        x1 = int(input[ptr])
        y1 = int(input[ptr+1])
        z1 = int(input[ptr+2])
        x2 = int(input[ptr+3])
        y2 = int(input[ptr+4])
        z2 = int(input[ptr+5])
        ptr += 6
        cuboids.append((x1, y1, z1, x2, y2, z2))
        x1_dict[x1].append(cuboid)
        x2_dict[x2].append(cuboid)
        y1_dict[y1].append(cuboid)
        y2_dict[y2].append(cuboid)
        z1_dict[z1].append(cuboid)
        z2_dict[z2].append(cuboid)
    
    for cuboid in cuboids:
        x1_i, y1_i, z1_i, x2_i, y2_i, z2_i = cuboid
        res = 0
        
        # Check x right
        for j in x1_dict.get(x2_i, []):
            jx1, jy1, jz1, jx2, jy2, jz2 = j
            if (jy1 < y2_i and jy2 > y1_i) and (jz1 < z2_i and jz2 > z1_i):
                res += 1
        
        # Check x left
        for j in x2_dict.get(x1_i, []):
            jx1, jy1, jz1, jx2, jy2, jz2 = j
            if (jy1 < y2_i and jy2 > y1_i) and (jz1 < z2_i and jz2 > z1_i):
                res += 1
        
        # Check y right
        for j in y1_dict.get(y2_i, []):
            jx1, jy1, jz1, jx2, jy2, jz2 = j
            if (jx1 < x2_i and jx2 > x1_i) and (jz1 < z2_i and jz2 > z1_i):
                res += 1
        
        # Check y left
        for j in y2_dict.get(y1_i, []):
            jx1, jy1, jz1, jx2, jy2, jz2 = j
            if (jx1 < x2_i and jx2 > x1_i) and (jz1 < z2_i and jz2 > z1_i):
                res += 1
        
        # Check z right
        for j in z1_dict.get(z2_i, []):
            jx1, jy1, jz1, jx2, jy2, jz2 = j
            if (jx1 < x2_i and jx2 > x1_i) and (jy1 < y2_i and jy2 > y1_i):
                res += 1
        
        # Check z left
        for j in z2_dict.get(z1_i, []):
            jx1, jy1, jz1, jx2, jy2, jz2 = j
            if (jx1 < x2_i and jx2 > x1_i) and (jy1 < y2_i and jy2 > y1_i):
                res += 1
        
        print(res)

if __name__ == '__main__':
    main()