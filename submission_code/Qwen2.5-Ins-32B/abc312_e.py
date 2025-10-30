import sys

def read_ints(): return map(int, sys.stdin.readline().strip().split())

def count_shared_faces(N, cuboids):
    # Initialize the count of shared faces for each cuboid
    shared_faces = [0] * N
    
    # Iterate over each cuboid
    for i in range(N):
        x1, y1, z1, x2, y2, z2 = cuboids[i]
        # Check for shared faces with other cuboids
        for j in range(N):
            if i == j:
                continue
            x1_j, y1_j, z1_j, x2_j, y2_j, z2_j = cuboids[j]
            # Check if the cuboids share a face
            if (x1 == x2_j and y1 <= y2_j and y2 >= y1_j and z1 <= z2_j and z2 >= z1_j) or \
               (x2 == x1_j and y1 <= y2_j and y2 >= y1_j and z1 <= z2_j and z2 >= z1_j) or \
               (y1 == y2_j and x1 <= x2_j and x2 >= x1_j and z1 <= z2_j and z2 >= z1_j) or \
               (y2 == y1_j and x1 <= x2_j and x2 >= x1_j and z1 <= z2_j and z2 >= z1_j) or \
               (z1 == z2_j and x1 <= x2_j and x2 >= x1_j and y1 <= y2_j and y2 >= y1_j) or \
               (z2 == z1_j and x1 <= x2_j and x2 >= x1_j and y1 <= y2_j and y2 >= y1_j):
                shared_faces[i] += 1
    
    return shared_faces

def main():
    N = int(input())
    cuboids = [read_ints() for _ in range(N)]
    shared_faces = count_shared_faces(N, cuboids)
    for count in shared_faces:
        print(count)

if __name__ == "__main__":
    main()