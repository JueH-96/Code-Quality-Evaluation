from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    cuboids = []
    same_x = defaultdict(int)
    same_y = defaultdict(int)
    same_z = defaultdict(int)
    starts_x = defaultdict(int)
    ends_x = defaultdict(int)
    starts_y = defaultdict(int)
    ends_y = defaultdict(int)
    starts_z = defaultdict(int)
    ends_z = defaultdict(int)
    
    for _ in range(N):
        x1 = int(input[ptr])
        y1 = int(input[ptr+1])
        z1 = int(input[ptr+2])
        x2 = int(input[ptr+3])
        y2 = int(input[ptr+4])
        z2 = int(input[ptr+5])
        ptr += 6
        cuboids.append((x1, y1, z1, x2, y2, z2))
        key = (x1, x2)
        same_x[key] += 1
        starts_x[x1] += 1
        ends_x[x2] += 1
        
        key_y = (y1, y2)
        same_y[key_y] += 1
        starts_y[y1] += 1
        ends_y[y2] += 1
        
        key_z = (z1, z2)
        same_z[key_z] += 1
        starts_z[z1] += 1
        ends_z[z2] += 1
    
    output = []
    for i in range(N):
        x1, y1, z1, x2, y2, z2 = cuboids[i]
        
        key_x = (x1, x2)
        sx = starts_x.get(x1, 0)
        ex = ends_x.get(x2, 0)
        sxex = same_x.get(key_x, 0)
        count_x = (sx + ex - sxex) - 1
        if count_x < 0:
            count_x = 0
        
        key_y = (y1, y2)
        sy = starts_y.get(y1, 0)
        ey = ends_y.get(y2, 0)
        syey = same_y.get(key_y, 0)
        count_y = (sy + ey - syey) - 1
        if count_y < 0:
            count_y = 0
        
        key_z = (z1, z2)
        sz = starts_z.get(z1, 0)
        ez = ends_z.get(z2, 0)
        szez = same_z.get(key_z, 0)
        count_z = (sz + ez - szez) - 1
        if count_z < 0:
            count_z = 0
        
        total = count_x + count_y + count_z
        output.append(str(total))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()