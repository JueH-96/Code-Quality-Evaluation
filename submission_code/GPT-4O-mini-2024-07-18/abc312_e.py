def count_shared_faces(cuboids):
    N = len(cuboids)
    face_count = [0] * N
    
    # Create a dictionary to track the faces of each cuboid
    face_map = {}
    
    for i in range(N):
        x1, y1, z1, x2, y2, z2 = cuboids[i]
        
        # Define the six faces of the cuboid
        faces = [
            (x1, y1, z1, x2, y1, z1),  # Face at z1
            (x1, y1, z2, x2, y1, z2),  # Face at z2
            (x1, y1, z1, x1, y2, z2),  # Face at x1
            (x2, y1, z1, x2, y2, z2),  # Face at x2
            (x1, y1, z1, x1, y1, z2),  # Face at y1
            (x1, y2, z1, x1, y2, z2),  # Face at y2
        ]
        
        for face in faces:
            if face not in face_map:
                face_map[face] = []
            face_map[face].append(i)
    
    # Count shared faces
    for face, indices in face_map.items():
        count = len(indices)
        for index in indices:
            face_count[index] += count - 1  # Exclude itself
    
    return face_count

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N = int(data[0])
    cuboids = []
    
    for i in range(1, N + 1):
        cuboid = tuple(map(int, data[i].split()))
        cuboids.append(cuboid)
    
    result = count_shared_faces(cuboids)
    
    for count in result:
        print(count)

if __name__ == "__main__":
    main()