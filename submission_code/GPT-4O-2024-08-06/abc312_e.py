# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    cuboids = []
    
    index = 1
    for _ in range(N):
        X1 = int(data[index])
        Y1 = int(data[index + 1])
        Z1 = int(data[index + 2])
        X2 = int(data[index + 3])
        Y2 = int(data[index + 4])
        Z2 = int(data[index + 5])
        cuboids.append((X1, Y1, Z1, X2, Y2, Z2))
        index += 6
    
    def share_face(c1, c2):
        X1, Y1, Z1, X2, Y2, Z2 = c1
        x1, y1, z1, x2, y2, z2 = c2
        
        # Check if they share a face on the x-axis
        if (X1 == x2 or X2 == x1) and (Y1 < y2 and Y2 > y1) and (Z1 < z2 and Z2 > z1):
            return True
        # Check if they share a face on the y-axis
        if (Y1 == y2 or Y2 == y1) and (X1 < x2 and X2 > x1) and (Z1 < z2 and Z2 > z1):
            return True
        # Check if they share a face on the z-axis
        if (Z1 == z2 or Z2 == z1) and (X1 < x2 and X2 > x1) and (Y1 < y2 and Y2 > y1):
            return True
        
        return False
    
    results = [0] * N
    
    for i in range(N):
        for j in range(i + 1, N):
            if share_face(cuboids[i], cuboids[j]):
                results[i] += 1
                results[j] += 1
    
    for result in results:
        print(result)