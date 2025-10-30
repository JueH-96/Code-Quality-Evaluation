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
        cuboids.append((x1, y1, z1, x2, y2, z2))
        index += 6
    
    # To store the count of adjacent cuboids for each cuboid
    adjacent_counts = [0] * N
    
    # We will check for each pair of cuboids if they share a face
    for i in range(N):
        for j in range(i+1, N):
            # Extract coordinates
            x1_i, y1_i, z1_i, x2_i, y2_i, z2_i = cuboids[i]
            x1_j, y1_j, z1_j, x2_j, y2_j, z2_j = cuboids[j]
            
            # Check if they share a face in x-direction
            if x2_i == x1_j or x2_j == x1_i:
                # Check if y and z ranges overlap
                if max(y1_i, y1_j) < min(y2_i, y2_j) and max(z1_i, z1_j) < min(z2_i, z2_j):
                    adjacent_counts[i] += 1
                    adjacent_counts[j] += 1
            # Check if they share a face in y-direction
            elif y2_i == y1_j or y2_j == y1_i:
                # Check if x and z ranges overlap
                if max(x1_i, x1_j) < min(x2_i, x2_j) and max(z1_i, z1_j) < min(z2_i, z2_j):
                    adjacent_counts[i] += 1
                    adjacent_counts[j] += 1
            # Check if they share a face in z-direction
            elif z2_i == z1_j or z2_j == z1_i:
                # Check if x and y ranges overlap
                if max(x1_i, x1_j) < min(x2_i, x2_j) and max(y1_i, y1_j) < min(y2_i, y2_j):
                    adjacent_counts[i] += 1
                    adjacent_counts[j] += 1
    
    # Print the results
    for count in adjacent_counts:
        print(count)

if __name__ == "__main__":
    main()