def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    
    cuboids = []
    for _ in range(N):
        x1 = int(data[idx])
        y1 = int(data[idx+1])
        z1 = int(data[idx+2])
        x2 = int(data[idx+3])
        y2 = int(data[idx+4])
        z2 = int(data[idx+5])
        cuboids.append((x1, x2, y1, y2, z1, z2))
        idx += 6
    
    # Initialize min and max maps for each dimension
    min_maps = [ [ [] for _ in range(101) ] for _ in range(3) ]
    max_maps = [ [ [] for _ in range(101) ] for _ in range(3) ]
    
    for i in range(N):
        x1, x2, y1, y2, z1, z2 = cuboids[i]
        # x dimension (d=0)
        min_maps[0][x1].append(i)
        max_maps[0][x2].append(i)
        # y dimension (d=1)
        min_maps[1][y1].append(i)
        max_maps[1][y2].append(i)
        # z dimension (d=2)
        min_maps[2][z1].append(i)
        max_maps[2][z2].append(i)
    
    result = []
    for i in range(N):
        count = 0
        x1, x2, y1, y2, z1, z2 = cuboids[i]
        # Check each dimension
        for d in range(3):
            # Check min face
            min_i = cuboids[i][2*d]
            for j in max_maps[d][min_i]:
                if j == i:
                    continue
                # Determine other two dimensions
                if d == 0:
                    a, b = 1, 2
                elif d == 1:
                    a, b = 0, 2
                else:
                    a, b = 0, 1
                # Check overlap in a and b dimensions
                ia1 = cuboids[i][2*a]
                ia2 = cuboids[i][2*a + 1]
                ja1 = cuboids[j][2*a]
                ja2 = cuboids[j][2*a + 1]
                overlap_a = max(ia1, ja1) < min(ia2, ja2)
                ib1 = cuboids[i][2*b]
                ib2 = cuboids[i][2*b + 1]
                jb1 = cuboids[j][2*b]
                jb2 = cuboids[j][2*b + 1]
                overlap_b = max(ib1, jb1) < min(ib2, jb2)
                if overlap_a and overlap_b:
                    count += 1
            
            # Check max face
            max_i = cuboids[i][2*d + 1]
            for j in min_maps[d][max_i]:
                if j == i:
                    continue
                if d == 0:
                    a, b = 1, 2
                elif d == 1:
                    a, b = 0, 2
                else:
                    a, b = 0, 1
                ia1 = cuboids[i][2*a]
                ia2 = cuboids[i][2*a + 1]
                ja1 = cuboids[j][2*a]
                ja2 = cuboids[j][2*a + 1]
                overlap_a = max(ia1, ja1) < min(ia2, ja2)
                ib1 = cuboids[i][2*b]
                ib2 = cuboids[i][2*b + 1]
                jb1 = cuboids[j][2*b]
                jb2 = cuboids[j][2*b + 1]
                overlap_b = max(ib1, jb1) < min(ib2, jb2)
                if overlap_a and overlap_b:
                    count += 1
        
        result.append(count)
    
    sys.stdout.write('
'.join(map(str, result)) + '
')

if __name__ == '__main__':
    main()