def main():
    import sys
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
    
    # To store the number of cuboids that share a face with each cuboid
    results = [0] * N
    
    # Check each pair of cuboids
    for i in range(N):
        x1, y1, z1, x2, y2, z2 = cuboids[i]
        for j in range(N):
            if i != j:
                x1j, y1j, z1j, x2j, y2j, z2j = cuboids[j]
                
                # Check if they share a face
                if (x2 == x1j and y1 < y2j and y2 > y1j and z1 < z2j and z2 > z1j) or \
                   (x1 == x2j and y1 < y2j and y2 > y1j and z1 < z2j and z2 > z1j) or \
                   (y2 == y1j and x1 < x2j and x2 > x1j and z1 < z2j and z2 > z1j) or \
                   (y1 == y2j and x1 < x2j and x2 > x1j and z1 < z2j and z2 > z1j) or \
                   (z2 == z1j and x1 < x2j and x2 > x1j and y1 < y2j and y2 > y1j) or \
                   (z1 == z2j and x1 < x2j and x2 > x1j and y1 < y2j and y2 > y1j):
                    results[i] += 1
    
    # Output results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()