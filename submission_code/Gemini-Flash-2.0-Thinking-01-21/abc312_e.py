def solve():
    n = int(input())
    cuboids_data = []
    for _ in range(n):
        x1, y1, z1, x2, y2, z2 = map(int, input().split())
        cuboids_data.append(((x1, y1, z1), (x2, y2, z2)))
    
    cuboid_ranges = []
    for (x1, y1, z1), (x2, y2, z2) in cuboids_data:
        cuboid_ranges.append({
            'x_min': min(x1, x2), 'x_max': max(x1, x2),
            'y_min': min(y1, y2), 'y_max': max(y1, y2),
            'z_min': min(z1, z2), 'z_max': max(z1, z2)
        })
        
    face_counts = [0] * n
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cuboid1 = cuboid_ranges[i]
            cuboid2 = cuboid_ranges[j]
            shares_face = False
            
            # Condition 1: x_max of i and x_min of j
            if cuboid1['x_max'] == cuboid2['x_min']:
                if max(cuboid1['y_min'], cuboid2['y_min']) < min(cuboid1['y_max'], cuboid2['y_max']) and \
                   max(cuboid1['z_min'], cuboid2['z_min']) < min(cuboid1['z_max'], cuboid2['z_max']):
                    shares_face = True
                    
            # Condition 2: x_min of i and x_max of j
            if not shares_face and cuboid1['x_min'] == cuboid2['x_max']:
                if max(cuboid1['y_min'], cuboid2['y_min']) < min(cuboid1['y_max'], cuboid2['y_max']) and \
                   max(cuboid1['z_min'], cuboid2['z_min']) < min(cuboid1['z_max'], cuboid2['z_max']):
                    shares_face = True
                    
            # Condition 3: y_max of i and y_min of j
            if not shares_face and cuboid1['y_max'] == cuboid2['y_min']:
                if max(cuboid1['x_min'], cuboid2['x_min']) < min(cuboid1['x_max'], cuboid2['x_max']) and \
                   max(cuboid1['z_min'], cuboid2['z_min']) < min(cuboid1['z_max'], cuboid2['z_max']):
                    shares_face = True
                    
            # Condition 4: y_min of i and y_max of j
            if not shares_face and cuboid1['y_min'] == cuboid2['y_max']:
                if max(cuboid1['x_min'], cuboid2['x_min']) < min(cuboid1['x_max'], cuboid2['x_max']) and \
                   max(cuboid1['z_min'], cuboid2['z_min']) < min(cuboid1['z_max'], cuboid2['z_max']):
                    shares_face = True
                    
            # Condition 5: z_max of i and z_min of j
            if not shares_face and cuboid1['z_max'] == cuboid2['z_min']:
                if max(cuboid1['x_min'], cuboid2['x_min']) < min(cuboid1['x_max'], cuboid2['x_max']) and \
                   max(cuboid1['y_min'], cuboid2['y_min']) < min(cuboid1['y_max'], cuboid2['y_max']):
                    shares_face = True
                    
            # Condition 6: z_min of i and z_max of j
            if not shares_face and cuboid1['z_min'] == cuboid2['z_max']:
                if max(cuboid1['x_min'], cuboid2['x_min']) < min(cuboid1['x_max'], cuboid2['x_max']) and \
                   max(cuboid1['y_min'], cuboid2['y_min']) < min(cuboid1['y_max'], cuboid2['y_max']):
                    shares_face = True
                    
            if shares_face:
                face_counts[i] += 1
                
    for count in face_counts:
        print(count)

if __name__ == '__main__':
    solve()