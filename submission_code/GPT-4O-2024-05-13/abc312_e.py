# YOUR CODE HERE
def count_shared_faces(N, cuboids):
    def is_face_shared(c1, c2):
        # Check if two cuboids share a face
        x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = c1
        x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = c2
        
        # Check for shared face on x-axis
        if (x1_max == x2_min or x1_min == x2_max) and \
           (y1_min < y2_max and y1_max > y2_min) and \
           (z1_min < z2_max and z1_max > z2_min):
            return True
        
        # Check for shared face on y-axis
        if (y1_max == y2_min or y1_min == y2_max) and \
           (x1_min < x2_max and x1_max > x2_min) and \
           (z1_min < z2_max and z1_max > z2_min):
            return True
        
        # Check for shared face on z-axis
        if (z1_max == z2_min or z1_min == z2_max) and \
           (x1_min < x2_max and x1_max > x2_min) and \
           (y1_min < y2_max and y1_max > y2_min):
            return True
        
        return False
    
    results = []
    for i in range(N):
        count = 0
        for j in range(N):
            if i != j and is_face_shared(cuboids[i], cuboids[j]):
                count += 1
        results.append(count)
    
    return results

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
cuboids = []
index = 1
for i in range(N):
    x1 = int(data[index])
    y1 = int(data[index + 1])
    z1 = int(data[index + 2])
    x2 = int(data[index + 3])
    y2 = int(data[index + 4])
    z2 = int(data[index + 5])
    cuboids.append((x1, y1, z1, x2, y2, z2))
    index += 6

results = count_shared_faces(N, cuboids)
for result in results:
    print(result)