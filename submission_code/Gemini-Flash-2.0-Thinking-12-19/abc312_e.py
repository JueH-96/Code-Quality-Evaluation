def solve():
    n = int(input())
    cuboids_coords = []
    for _ in range(n):
        coords = list(map(int, input().split()))
        cuboids_coords.append(coords)
    
    cuboids = []
    for coords in cuboids_coords:
        x1, y1, z1, x2, y2, z2 = coords
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        zmin = min(z1, z2)
        zmax = max(z1, z2)
        cuboids.append({'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax})
        
    face_counts = [0] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            cuboid1 = cuboids[i]
            cuboid2 = cuboids[j]
            shares_face = False
            
            # Condition 1: Right face of i and left face of j
            if cuboid1['xmax'] == cuboid2['xmin']:
                if max(cuboid1['ymin'], cuboid2['ymin']) < min(cuboid1['ymax'], cuboid2['ymax']) and \
                   max(cuboid1['zmin'], cuboid2['zmin']) < min(cuboid1['zmax'], cuboid2['zmax']):
                    shares_face = True
                    
            # Condition 2: Left face of i and right face of j
            if not shares_face and cuboid1['xmin'] == cuboid2['xmax']:
                if max(cuboid1['ymin'], cuboid2['ymin']) < min(cuboid1['ymax'], cuboid2['ymax']) and \
                   max(cuboid1['zmin'], cuboid2['zmin']) < min(cuboid1['zmax'], cuboid2['zmax']):
                    shares_face = True
                    
            # Condition 3: Top face of i and bottom face of j
            if not shares_face and cuboid1['ymax'] == cuboid2['ymin']:
                if max(cuboid1['xmin'], cuboid2['xmin']) < min(cuboid1['xmax'], cuboid2['xmax']) and \
                   max(cuboid1['zmin'], cuboid2['zmin']) < min(cuboid1['zmax'], cuboid2['zmax']):
                    shares_face = True
                    
            # Condition 4: Bottom face of i and top face of j
            if not shares_face and cuboid1['ymin'] == cuboid2['ymax']:
                if max(cuboid1['xmin'], cuboid2['xmin']) < min(cuboid1['xmax'], cuboid2['xmax']) and \
                   max(cuboid1['zmin'], cuboid2['zmin']) < min(cuboid1['zmax'], cuboid2['zmax']):
                    shares_face = True
                    
            # Condition 5: Back face of i and front face of j
            if not shares_face and cuboid1['zmax'] == cuboid2['zmin']:
                if max(cuboid1['xmin'], cuboid2['xmin']) < min(cuboid1['xmax'], cuboid2['xmax']) and \
                   max(cuboid1['ymin'], cuboid2['ymin']) < min(cuboid1['ymax'], cuboid2['ymax']):
                    shares_face = True
                    
            # Condition 6: Front face of i and back face of j
            if not shares_face and cuboid1['zmin'] == cuboid2['zmax']:
                if max(cuboid1['xmin'], cuboid2['xmin']) < min(cuboid1['xmax'], cuboid2['xmax']) and \
                   max(cuboid1['ymin'], cuboid2['ymin']) < min(cuboid1['ymax'], cuboid2['ymax']):
                    shares_face = True
                    
            if shares_face:
                face_counts[i] += 1
                face_counts[j] += 1
                
    print(*(face_counts))

if __name__ == '__main__':
    solve()