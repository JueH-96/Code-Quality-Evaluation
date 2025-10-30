import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    cuboids = []
    for _ in range(N):
        x1 = int(input[idx])
        y1 = int(input[idx+1])
        z1 = int(input[idx+2])
        x2 = int(input[idx+3])
        y2 = int(input[idx+4])
        z2 = int(input[idx+5])
        idx +=6
        cuboids.append( (x1,y1,z1, x2,y2,z2) )
    
    groups = defaultdict(list)
    for i in range(N):
        x1,y1,z1,x2,y2,z2 = cuboids[i]
        groups[('x', x1, 'lower')].append(i)
        groups[('x', x2, 'upper')].append(i)
        groups[('y', y1, 'lower')].append(i)
        groups[('y', y2, 'upper')].append(i)
        groups[('z', z1, 'lower')].append(i)
        groups[('z', z2, 'upper')].append(i)
    
    ans = [0] * N
    
    for i in range(N):
        x1, y1, z1, x2, y2, z2 = cuboids[i]
        count = 0
        
        # X lower face
        key = ('x', x1, 'upper')
        list_j = groups.get(key, [])
        for j in list_j:
            if i == j:
                continue
            x1j, y1j, z1j, x2j, y2j, z2j = cuboids[j]
            overlap_y = (y1 < y2j) and (y1j < y2)
            overlap_z = (z1 < z2j) and (z1j < z2)
            if overlap_y and overlap_z:
                count += 1
        
        # X upper face
        key = ('x', x2, 'lower')
        list_j = groups.get(key, [])
        for j in list_j:
            if i == j:
                continue
            x1j, y1j, z1j, x2j, y2j, z2j = cuboids[j]
            overlap_y = (y1 < y2j) and (y1j < y2)
            overlap_z = (z1 < z2j) and (z1j < z2)
            if overlap_y and overlap_z:
                count += 1
        
        # Y lower face
        key = ('y', y1, 'upper')
        list_j = groups.get(key, [])
        for j in list_j:
            if i == j:
                continue
            x1j, y1j, z1j, x2j, y2j, z2j = cuboids[j]
            overlap_x = (x1 < x2j) and (x1j < x2)
            overlap_z = (z1 < z2j) and (z1j < z2)
            if overlap_x and overlap_z:
                count += 1
        
        # Y upper face
        key = ('y', y2, 'lower')
        list_j = groups.get(key, [])
        for j in list_j:
            if i == j:
                continue
            x1j, y1j, z1j, x2j, y2j, z2j = cuboids[j]
            overlap_x = (x1 < x2j) and (x1j < x2)
            overlap_z = (z1 < z2j) and (z1j < z2)
            if overlap_x and overlap_z:
                count += 1
        
        # Z lower face
        key = ('z', z1, 'upper')
        list_j = groups.get(key, [])
        for j in list_j:
            if i == j:
                continue
            x1j, y1j, z1j, x2j, y2j, z2j = cuboids[j]
            overlap_x = (x1 < x2j) and (x1j < x2)
            overlap_y = (y1 < y2j) and (y1j < y2)
            if overlap_x and overlap_y:
                count += 1
        
        # Z upper face
        key = ('z', z2, 'lower')
        list_j = groups.get(key, [])
        for j in list_j:
            if i == j:
                continue
            x1j, y1j, z1j, x2j, y2j, z2j = cuboids[j]
            overlap_x = (x1 < x2j) and (x1j < x2)
            overlap_y = (y1 < y2j) and (y1j < y2)
            if overlap_x and overlap_y:
                count += 1
        
        ans[i] = count
    
    for a in ans:
        print(a)

if __name__ == '__main__':
    main()