# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    cuboids = []
    index = 1
    for _ in range(N):
        x1 = int(data[index])
        y1 = int(data[index+1])
        z1 = int(data[index+2])
        x2 = int(data[index+3])
        y2 = int(data[index+4])
        z2 = int(data[index+5])
        cuboids.append(((x1, y1, z1), (x2, y2, z2)))
        index += 6
    
    # To count the number of cuboids sharing a face with each cuboid
    # We need to check for each pair of cuboids if they share a face
    # Since N can be up to 1e5, we need an efficient way to do this
    # We can use a dictionary to map each face to the cuboids that have it
    
    # We will represent each face as a tuple (type, coordinates)
    # type: 0 for x=constant, 1 for y=constant, 2 for z=constant
    # coordinates: (x, y1, z1, y2, z2) for x=constant, etc.
    
    face_to_cuboids = {}
    
    for i in range(N):
        (x1, y1, z1), (x2, y2, z2) = cuboids[i]
        # x=constant faces
        face1 = (0, x1, y1, z1, y2, z2)
        face2 = (0, x2, y1, z1, y2, z2)
        # y=constant faces
        face3 = (1, y1, x1, z1, x2, z2)
        face4 = (1, y2, x1, z1, x2, z2)
        # z=constant faces
        face5 = (2, z1, x1, y1, x2, y2)
        face6 = (2, z2, x1, y1, x2, y2)
        
        for face in [face1, face2, face3, face4, face5, face6]:
            if face not in face_to_cuboids:
                face_to_cuboids[face] = []
            face_to_cuboids[face].append(i)
    
    # Now, for each cuboid, count the number of other cuboids sharing a face
    result = [0] * N
    for face in face_to_cuboids:
        cuboid_list = face_to_cuboids[face]
        if len(cuboid_list) >= 2:
            for i in cuboid_list:
                result[i] += len(cuboid_list) - 1
    
    # Since each face is shared by two cuboids, we need to divide by 2
    # But since each cuboid is counted once for each face it shares, we need to adjust
    # Wait, no. Each face is shared by two cuboids, so each cuboid is counted once for each face it shares
    # So the result is correct as is
    
    for res in result:
        print(res)

if __name__ == "__main__":
    main()