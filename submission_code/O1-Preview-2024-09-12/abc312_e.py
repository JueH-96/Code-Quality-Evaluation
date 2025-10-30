# YOUR CODE HERE
import sys
import threading
from collections import Counter

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    idx = 1
    N = int(N)

    cuboids = []
    for _ in range(N):
        X1 = int(N_and_rest[idx])
        Y1 = int(N_and_rest[idx+1])
        Z1 = int(N_and_rest[idx+2])
        X2 = int(N_and_rest[idx+3])
        Y2 = int(N_and_rest[idx+4])
        Z2 = int(N_and_rest[idx+5])
        idx +=6
        cuboids.append((X1, Y1, Z1, X2, Y2, Z2))

    face_counter = Counter()
    face_keys_per_cuboid = []

    for i in range(N):
        X1, Y1, Z1, X2, Y2, Z2 = cuboids[i]
        faces = []
        
        # Faces perpendicular to x-axis
        y_range = (min(Y1, Y2), max(Y1, Y2))
        z_range = (min(Z1, Z2), max(Z1, Z2))
        
        faces.append(('x', X1, y_range, z_range))
        faces.append(('x', X2, y_range, z_range))
        
        # Faces perpendicular to y-axis
        x_range = (min(X1, X2), max(X1, X2))
        z_range = (min(Z1, Z2), max(Z1, Z2))
        
        faces.append(('y', Y1, x_range, z_range))
        faces.append(('y', Y2, x_range, z_range))
        
        # Faces perpendicular to z-axis
        x_range = (min(X1, X2), max(X1, X2))
        y_range = (min(Y1, Y2), max(Y1, Y2))
        
        faces.append(('z', Z1, x_range, y_range))
        faces.append(('z', Z2, x_range, y_range))
        
        face_keys_per_cuboid.append(faces)
        
        for face in faces:
            face_counter[face] +=1

    # Now compute counts for each cuboid
    results = []
    for faces in face_keys_per_cuboid:
        count = 0
        for face in faces:
            c = face_counter[face]
            count += c -1  # Number of other cuboids sharing this face
        results.append(count)

    for res in results:
        print(res)

threading.Thread(target=main).start()